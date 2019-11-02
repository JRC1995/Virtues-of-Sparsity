import os.path
import numpy as np
import random
import string
import pickle
import math


# %%

train_filepath = "../Data/train.txt"
val_filepath = '../Data/valid.txt'
test_filepath = '../Data/test.txt'

# %%

counter = {}
tags2idx = {}

word_vec_dim = 300


def loadCoNLL2003(filename, test=False):

    sequences = []
    tag_sequences = []
    sequence = []
    tag_sequence = []
    global counter
    global tags2idx

    file = open(filename, 'r')

    i = 0
    for line in file.readlines():

        if i > 1 and line in ['\n', '\r\n']:
            if sequence:
                sequences.append(sequence)
                tag_sequences.append(tag_sequence)
                sequence = []
                tag_sequence = []

        elif i > 1:

            row = line.strip().split(' ')

            word = row[0].lower()
            tag = row[-1]

            if tag not in tags2idx:
                tags2idx[tag] = len(tags2idx)

            if word != "-docstart-":

                if not test:

                    if word not in counter:
                        counter[word] = 1
                    else:
                        counter[word] += 1

                sequence.append(word)
                tag_sequence.append(tag)
        i += 1

    file.close()

    return sequences, tag_sequences


# %%
train_sequences, train_tag_sequences = loadCoNLL2003(train_filepath)
val_sequences, val_tag_sequences = loadCoNLL2003(val_filepath)
test_sequences, test_tag_sequences = loadCoNLL2003(test_filepath)

# %%

special_tags = ['<UNK>', '<PAD>']

vocab = [word for word in counter]
vocab += special_tags


def loadEmbeddings(filename):
    vocab2embd = {}

    with open(filename) as infile:
        for line in infile:
            row = line.strip().split(' ')
            word = row[0].lower()
            # print(word)
            if word not in vocab2embd:
                vec = np.asarray(row[1:], np.float32)
                if len(vec) == word_vec_dim:
                    vocab2embd[word] = vec

    print('Embedding Loaded.')
    return vocab2embd


vocab2embd = loadEmbeddings('../../Embeddings/glove.840B.300d.txt')

print("FULL GLOVE VOCAB LEN:", len(vocab2embd))

cutoff = 10

for word in vocab:
    if word not in vocab2embd and word not in special_tags:
        if counter[word] >= cutoff:
            vocab2embd[word] = np.random.uniform(-math.sqrt(3/word_vec_dim),
                                                 +math.sqrt(3/word_vec_dim),
                                                 (word_vec_dim))

vocab2embd['<PAD>'] = np.zeros(word_vec_dim, np.float32)
vocab2embd['<UNK>'] = np.random.uniform(-math.sqrt(3/word_vec_dim),
                                        +math.sqrt(3/word_vec_dim),
                                        (word_vec_dim))


embd = []
vocab2idx = {}


for idx, word in enumerate(vocab):
    if word in vocab2embd:
        vocab2idx[word] = len(vocab2idx)
        embd.append(vocab2embd[word])


embd = np.asarray(embd, np.float32)

print("Embeddings Shape: ", embd.shape)

print("Vocab Length: {}".format(len(vocab2idx)))

print("Tags2idx dictionary: ", tags2idx)

# FIX BIO LABELS


def fixBIO(tag_sequences):

    fixed_tag_sequences = []

    for tag_sequence in tag_sequences:

        fixed_tag_sequence = []

        for i, tag in enumerate(tag_sequence):

            tag_prefix = tag[0]
            fixed_tag = tag

            if i == 0:
                if tag_prefix == "I":
                    fixed_tag = tag.replace('I-', 'B-')
            else:
                if tag_prefix == "I":
                    prev_tag = tag_sequence[i-1]
                    prev_tag_prefix = prev_tag[0]
                    if prev_tag_prefix == "O":
                        fixed_tag = tag.replace('I-', 'B-')

            fixed_tag_sequence.append(fixed_tag)

        fixed_tag_sequences.append(fixed_tag_sequence)

    return fixed_tag_sequences


train_tag_sequences = fixBIO(train_tag_sequences)
val_tag_sequences = fixBIO(val_tag_sequences)
test_tag_sequences = fixBIO(test_tag_sequences)


# %%

n = 5
train_idx = [random.randint(0, len(train_sequences)) for i in range(n)]
val_idx = [random.randint(0, len(val_sequences)) for i in range(n)]
test_idx = [random.randint(0, len(test_sequences)) for i in range(n)]

print("\nExample Training Data:\n")

for i in train_idx:
    print('Sentence: ', end='')
    print(train_sequences[i])
    print('\nTags: ', end=' ')
    print(train_tag_sequences[i], end='\n\n')

print("\nExample Validation Data:\n")

for i in val_idx:
    print('Sentence: ', end='')
    print(val_sequences[i])
    print('\nTags: ', end=' ')
    print(val_tag_sequences[i], end='\n\n')

print("\nExample Test Data:\n")

for i in test_idx:
    print('Sentence: ', end='')
    print(test_sequences[i])
    print('\nTags: ', end=' ')
    print(test_tag_sequences[i], end='\n\n')


def vectorize(sequences):
    global vocab2idx
    sequences_vec = []
    for sequence in sequences:
        sequence_vec = [vocab2idx.get(word, vocab2idx['<UNK>']) for word in sequence]
        sequences_vec.append(sequence_vec)

    return sequences_vec


train_sequences = vectorize(train_sequences)
val_sequences = vectorize(val_sequences)
test_sequences = vectorize(test_sequences)


def vectorize_tags(tag_sequences):
    global tags2idx
    tag_sequences_vec = []
    for tag_sequence in tag_sequences:
        tag_sequence_vec = [tags2idx[tag] for tag in tag_sequence]
        tag_sequences_vec.append(tag_sequence_vec)

    return tag_sequences_vec


train_tag_sequences = vectorize_tags(train_tag_sequences)
val_tag_sequences = vectorize_tags(val_tag_sequences)
test_tag_sequences = vectorize_tags(test_tag_sequences)

print("\n\nAFTER VECTORIZATION\n\n")


print("\nExample Training Data:\n")

for i in train_idx:
    print('Sentence: ', end='')
    print(train_sequences[i])
    print('\nTags: ', end=' ')
    print(train_tag_sequences[i], end='\n\n')

print("\nExample Validation Data:\n")

for i in val_idx:
    print('Sentence: ', end='')
    print(val_sequences[i])
    print('\nTags: ', end=' ')
    print(val_tag_sequences[i], end='\n\n')

print("\nExample Test Data:\n")

for i in test_idx:
    print('Sentence: ', end='')
    print(test_sequences[i])
    print('\nTags: ', end=' ')
    print(test_tag_sequences[i], end='\n\n')

pickle_dict = {'tags2idx': tags2idx,
               'train_text': train_sequences,
               'train_tags': train_tag_sequences,
               'val_text': val_sequences,
               'val_tags': val_tag_sequences,
               'test_text': test_sequences,
               'test_tags': test_tag_sequences,
               'vocab2idx': vocab2idx,
               'embd': embd}

with open('../Processed_Data/CoNLL_NER_2003.pkl', 'wb') as fp:
    pickle.dump(pickle_dict, fp)
