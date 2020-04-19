import re
import torch
import itertools
import unicodedata
import numpy as np

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
MAX_LENGTH = 100
MIN_COUNT = 5    # Minimum word count threshold for trimming

def levenshtein(source, target):
    source = ' ' + unicodedata.normalize('NFD',source)
    target = unicodedata.normalize('NFD',target)
    if len(source) < len(target):
        return levenshtein(target, source)
    if len(target) == 0:
        return len(source)
    source = np.array(tuple(source))
    target = np.array(tuple(target))
    previous_row = np.arange(target.size + 1)
    for s in source:
        current_row = previous_row + 1
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)
        previous_row = current_row
    return previous_row[-1]

def clean_abbr(sentence):
    sentence = sentence.lower()
    char_remove = '-.'
    and_flag = -1
    start_pos = sentence.find('~')
    end_pos = sentence.find('#')
    abbr = sentence[start_pos+2:end_pos-1]
    for char in char_remove:
        abbr = abbr.replace(char,'')
    if '&' in abbr:
        and_flag = abbr.find('&')
        abbr = abbr.replace('&','')
    sen_out = sentence[:start_pos+2] + abbr + sentence[end_pos-1:]
    return sen_out,abbr,and_flag

def insert_va(expand,pos):
    if pos <= 0:
        return expand
    count = 0
    expand = ' ' + expand
    expand = expand.replace('  ',' ')
    for i in range(len(expand)):
        if expand[i] == ' ': 
            count += 1
        if count == pos+1:
            pos = i
            break
    result = expand[:pos] + ' và ' + expand[pos:]
    result = result.replace('  ',' ')
    return result


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in list(input_sentence):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in list(output_sentence):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

def normalizeString(s):
    # Tách dấu câu nếu kí tự liền nhau
    marks = '[.!?,-${}()]'
    r = "(["+"\\".join(marks)+"])"
    s = re.sub(r, r" \1 ", s)
    # Thay thế nhiều spaces bằng 1 space.
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def _data_read(fn):
  with open(fn, 'r') as fn:
    train = fn.readlines()
  train = [item[:-1].lower() for item in train]  
  return train

def filterPair(p):
    # Input sequences need to preserve the last word for EOS token
    return len(list(p[0])) < MAX_LENGTH and len(list(p[1])) < MAX_LENGTH

# Filter pairs using filterPair condition
def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# # Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(voc, pairs):
    print("Read {!s} sentence pairs".format(len(pairs)))
    # pairs = filterPairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

def get_target(input_batch,start_nsw_index,end_nsw_index):
    output = []
    input_batch_revert = input_batch.t()
    for i in range(len(input_batch_revert)):
        tmp = input_batch_revert[i]
        start = (tmp == start_nsw_index).nonzero()
        end = (tmp == end_nsw_index).nonzero()
        output.append(list(tmp[start+2:end-1]))
    return output        


class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3  # Count SOS, EOS, PAD

    def addSentence(self, sentence):
        for word in list(sentence):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens

        for word in keep_words:
            self.addWord(word)

def readVocs(lines, corpus_name = 'corpus'):
    # Split every line into pairs and normalize
    pairs = [[normalizeString(str(s)) for s in l] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in list(sentence)] + [EOS_token]

# Padding thêm 0 vào list nào có độ dài nhỏ hơn về phía bên phải
def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))
  
# Tạo ma trận binary có kích thước như ma trận truyền vào l nhưng giá trị của mỗi phần tử đánh dấu 1 hoặc 0 tương ứng với padding hoặc không padding
def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(list(x[0])), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len
