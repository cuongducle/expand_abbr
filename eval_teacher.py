import numpy as np
import re
import codecs
# import torchtext
# from torchtext.data import Field
from utils import Voc,get_target,normalizeString,indexesFromSentence,insert_va,levenshtein,clean_abbr
from model import EncoderRNN,LuongAttnDecoderRNN
import torch
import torch.nn as nn
import unicodedata
import time
import pickle
import pickle

with open('dic.pkl','rb') as f:
    dic = pickle.load(f)  
loadFilename = "300000_checkpoint.tar"
USE_CUDA = torch.cuda.is_available()
#device = torch.device("cuda" if USE_CUDA else "cpu")
device = torch.device("cpu")
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

corpus_name = 'expand_abbr' 
voc = Voc(corpus_name)
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 100
MAX_LENGTH = 200

def evaluate(sentence, max_length=MAX_LENGTH):
    time_start = time.time()
    sentence = normalizeString(sentence)
    sentence = unicodedata.normalize('NFD',sentence)
    indexes_batch = [indexesFromSentence(voc, sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    tokens, score = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    result = ''
    for char in decoded_words:
        if char != 'EOS':
            result += char
        else:
            break
    time_pred = time.time() - time_start
    return result,torch.sum(score)/len(result),time_pred


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        reference = get_target(input_seq,start_nsw,end_nsw)[0]
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            tmp = decoder_input[0]
            if tmp == 2:
                return all_tokens, all_scores
            if tmp == voc.word2index[' ']:
                if (len(reference) >= 1):
                    decoder_input = reference.pop(0).item()
                    if decoder_input == w_char:
                        decoder_input = u_char
                else:
                    decoder_input = 2 #### EOS
                decoder_input = torch.tensor([decoder_input],device=device)
            else:
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        return all_tokens, all_scores

if loadFilename:
    #checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']
    
blank = voc.word2index[' ']
w_char = voc.word2index['w']
u_char = voc.word2index['u']
start_nsw = voc.word2index['~']
end_nsw = voc.word2index['#']

embedding = nn.Embedding(voc.num_words, hidden_size)
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder.eval()
decoder.eval()
searcher = GreedySearchDecoder(encoder, decoder)


def expand(sentence):
    sen,abbr,and_pos = clean_abbr(sentence)
    expand = ''
    if len(dic[abbr]) == 0:
        return "null",-1
    if len(dic[abbr]) == 1:
        expand = dic[abbr][0]
        expand = insert_va(expand,and_pos)
        return expand,0
    if len(dic[abbr]) >= 2:
        pred,score,time = evaluate(sen)
        tmp = len(pred)
        for item in dic[abbr]:
            if levenshtein(item,pred) < tmp:
                expand = item
                tmp = levenshtein(item,pred)
        if tmp > 2:
            expand = "n
        expand = insert_va(expand,and_pos)
        return expand,score.item()

# print(expand('cô giáo tôi là ~ th.s #'))

if __name__ == "__main__":
    data = []
    label = []
    with open('sen_val.txt','r') as f:
        for line in f:
            line = line.replace('\n','')
            data.append(line)

    with open('extend_val.txt','r') as f:
        for line in f:
            line = line.replace('\n','')
            label.append(line)

    ############
    a = time.time()
    count = 0
    num_sen = 30
    sentence_wrong = []
    expand_wrong = []
    label_wrong = []
    score_wrong = []
    for i in range(num_sen):
        sentence = data[i]
        print('sentence : ',sentence)
        pred,score = evaluate(sentence)
        pred = unicodedata.normalize('NFC',pred)
        label[i] = unicodedata.normalize('NFC',label[i])[:-1]
        if (pred == label[i]):
            count += 1
        else:
            print('sai roi /////////////////////////////////////////////////')
            sentence_wrong.append(sentence)
            expand_wrong.append(pred)
            label_wrong.append(label_wrong)
            score_wrong.append(score)

        print('predict  :',pred)
        print('label    :',label[i])
        print('score    :',torch.sum(score))
        # print(list(label[i]))
        # print(list(pred))
        print('-----------')
        print()


    print('time  : ',(time.time() - a)/num_sen)
    print('score :', count/num_sen)
    print('num_sen :', num_sen)

    for i in range(len(sentence_wrong)):
        print('sen_wrong   :' , sentence_wrong[i])
        print('pred_wrong  :' , expand_wrong[i])
        print('label_wrong :' , label_wrong[i])
        print('score_wrong :',torch.sum(score_wrong[i]))
        print( '-------------------------')
        print()
