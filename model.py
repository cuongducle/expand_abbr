from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        # set bidirectional = True for bidirectional
        # https://pytorch.org/docs/stable/nn.html?highlight=gru#torch.nn.GRU to get more information
        self.gru = nn.GRU(input_size = hidden_size, # number of expected feature of input x 
                          hidden_size = hidden_size, # number of expected feature of hidden state 
                          num_layers = n_layers, # number of GRU layers
                          dropout=(0 if n_layers == 1 else dropout), # dropout probability apply in encoder network
                          bidirectional=True # one or two directions.
                         )

    def forward(self, input_seq, input_lengths, hidden=None):
        # Step 1: Convert word indexes to embeddings
        # shape: (max_length , batch_size , hidden_size)
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module. Padding zero when length less than max_length of input_lengths.
        # shape: (max_length , batch_size , hidden_size)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Step 2: Forward packed through GRU
        # outputs is output of final GRU layer
        # hidden is concatenate of all hidden states corresponding with each time step.
        # outputs shape: (max_length , batch_size , hidden_size x num_directions)
        # hidden shape: (n_layers x num_directions , batch_size , hidden_size)
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding. Revert of pack_padded_sequence
        # outputs shape: (max_length , batch_size , hidden_size x num_directions)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs to reshape shape into (max_length , batch_size , hidden_size)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # outputs shape:(max_length , batch_size , hidden_size)
        # hidden shape: (n_layers x num_directions , batch_size , hidden_size)
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        # encoder_output shape:(max_length , batch_size , hidden_size)
        # hidden shape: (1 , batch_size , hidden_size)
        # return shape: (max_length, batch_size)
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        # encoder_output shape:(max_length , batch_size , hidden_size)
        # hidden shape: (batch_size , hidden_size)
        # energy shape: (max_length , batch_size , hidden_size)
        # return shape: (max_length , batch_size)
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        # encoder_output shape:(max_length , batch_size , hidden_size)
        # hidden shape: (batch_size , hidden_size)
        # energy shape: (max_length , batch_size , 2*hidden_size)
        # self.v shape: (hidden_size)
        # return shape: (max_length , batch_size)
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        # attn_energies.shape: (max_length , batch_size)
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        # attn_energies.shape: (batch_size , max_length)
        attn_energies = attn_energies.t()
        # Return the softmax normalized probability scores (with added dimension)
        attn_weights = F.softmax(attn_energies, dim=1).unsqueeze(1)
        # attn_weights shape: (batch_size , 1 , max_length)
        return attn_weights


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        '''
        input_step: list time step index of batch. shape (1 x batch_size)
        last_hidden: last hidden output of hidden layer (we can take in right direction or left direction upon us) which have shape = (n_layers x batch_size x hidden_size)
        encoder_outputs: output of encoder 
        '''
        #===========================================
        # Step 1: Embedding current sequence index
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        # embedded shape: 1 x batch_size x hidden_size
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        
        #===========================================
        # Step 2: pass embedded and last hidden into decoder
        # Forward through unidirectional GRU
        # rnn_output shape: 1 x batch_size x hidden_size
        # hidden shape: n_layers x batch_size x hidden_size
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        # attn_weights shape: batch_size x 1 x max_length
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        # encoder_outputs shape: max_length x batch_size x hidden_size
        # context shape: batch_size x 1 x hidden_size
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        # rnn_output shape: batch_size x hidden_size
        rnn_output = rnn_output.squeeze(0)
        # context shape: batch_size x hidden_size
        context = context.squeeze(1)
        
        #===========================================
        # Step 3: calculate output probability distribution 
        # concat_input shape: batch_size x (2*hidden_size)
        concat_input = torch.cat((rnn_output, context), 1)
        # concat_output shape: batch_size x hidden_size
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        # output shape: output_size
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
