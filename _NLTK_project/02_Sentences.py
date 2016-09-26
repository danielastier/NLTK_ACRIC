__author__ = 'Daniela Stier'

### IMPORT STATEMENTS
import string
import pandas as pd
import numpy as np
from collections import defaultdict

# read preprocessed data, store in lists (clean_data)
reader = open("data_pre_processed.txt", 'r')
labels = list()
clean_data = list()
for line in reader.readlines():
    line_split = line.split("\t")
    # if line_split[0] == "norm":
    #     labels.append(0)
    # elif line_split[0] == "narc":
    #     labels.append(1)
    # elif line_split[0] == "psych":
    #     labels.append(2)
    labels.append(line_split[0])
    sentence = list()
    for sent in line_split[1:]:
        sent_split = sent.split(", ")
        temp = list()
        for tagged_word in sent_split:
            if tagged_word.startswith("['"):
                temp.append(tagged_word[2:-1])
            elif tagged_word.endswith("']"):
                temp.append(tagged_word[1:-2])
            else:
                temp.append(tagged_word[1:-1])
        if len(temp) > 1:
            sentence.append(temp)
    clean_data.append(sentence)

header = list()
header.append("class")

# clean data excluding punctuation marks
puncts = string.punctuation
clean_data = [[[token for token in sent if not token[0:token.index("/")].lower() in puncts] for sent in text] for text in clean_data]

# list of words only (excluding pos-tags)
pure_words = [[[word[0:word.index("/")] for word in sent] for sent in text] for text in clean_data]

# number of words
num_words = [[len(sent) for sent in text] for text in pure_words]
num_words_summed = [sum(sent) for sent in num_words]

# number of characters
num_chars = [[[len(word) for word in sent] for sent in text] for text in pure_words]
num_chars_summed_sent = [[sum(sent) for sent in text] for text in num_chars]
num_chars_summed = [sum(text) for text in num_chars_summed_sent]


######### MEASURES CONCERNING SENTENCES ##############################

# number of sentences, output: int(number)
num_sentences = [len(text) for text in clean_data]
header.append("num_sent")

# average sentence length (according to the whole text, based on words and characters), output: float(number)
# dividing the total number of words|chars in a text by the total number of sentences
av_sentence_length_words = list()
for i, j in zip(num_words_summed, num_sentences):
    if j > 0:
        av_sentence_length_words.append(i/j)
    else:
        av_sentence_length_words.append(0)
#av_sentence_length_words = [(num_word/num_sent) for num_word, num_sent in zip(num_words_summed, num_sentences)]
av_sentence_length_chars = list()
for i, j in zip(num_chars_summed, num_sentences):
    if j > 0:
        av_sentence_length_chars.append(i/j)
    else:
        av_sentence_length_chars.append(0)
#av_sentence_length_chars = [(num_char/num_sent) for num_char, num_sent in zip(num_chars_summed, num_sentences)]
header.append("av_sent_len_words")
header.append("av_sent_len_chars")

# n-word sentences distribution (according to the whole text, based on words), output: float(number)
# rel. freq of each sentence-length: dividing total number of sentences of that length by total number of sentences
max_num_word = [np.mean(sent) for sent in num_words]
print(max_num_word)
#max_num_word = np.mean([max(doc) for doc in num_words if len(doc) > 0])
counter = defaultdict(int)
c = 0
for i in range(int(max_num_word[c])-15, int(max_num_word[c])+15):
#for i in range(1, int(max_num_word)+25):
    counter[i] = [text.count(i) for text in num_words]
    header.append(str("sent_len_dist_words" + str(i)))
    c += 1

sentence_length_dist_words = list()
for value in counter.values():
    temp = list()
    for i in range(len(num_sentences)):
        if num_sentences[i] > 0:
            temp.append(value[i]/num_sentences[i])
        else:
            temp.append(0)
    sentence_length_dist_words.append(temp)

# n-character sentences distribution (according to the whole text, based on characters)), output: float(number)
# rel. freq of each sentence-length: dividing total number of sentences of that length by total number of sentences
max_num_char = [np.mean(sent) for sent in num_chars_summed_sent]
print(max_num_char)
#max_num_char = np.mean([max([sent for sent in doc]) for doc in num_chars_summed_sent if len(doc) > 0])
counter = defaultdict(int)
c = 0
for i in range(int(max_num_char[c])-60, int(max_num_char[c])+60):
    counter[i] = [text.count(i) for text in num_chars_summed_sent]
    header.append(str("sent_len_dist_chars" + str(i)))
    c += 1

sentence_length_dist_chars = list()
for value in counter.values():
    temp = list()
    for i in range(len(num_sentences)):
        if num_sentences[i] > 0:
            temp.append(value[i]/num_sentences[i])
        else:
            temp.append(0)
    sentence_length_dist_chars.append(temp)

# store vectorized data in 'test_data_sent_vector.csv' file, including labels at first position
data_matrix = pd.concat([pd.DataFrame(labels), pd.DataFrame(num_sentences), pd.DataFrame(av_sentence_length_words), pd.DataFrame(av_sentence_length_chars), pd.DataFrame(sentence_length_dist_words).transpose(), pd.DataFrame(sentence_length_dist_chars).transpose()], axis=1)
#data_matrix.to_csv('_data_vectors/sent_vector.csv', index=False, delimiter=',', header=header)
#data_matrix.to_csv('_data_vectors_wider/sent_vector.csv', index=False, delimiter=',', header=header)