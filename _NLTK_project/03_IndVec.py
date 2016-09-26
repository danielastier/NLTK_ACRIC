__author__ = 'Daniela Stier'

# import statements
from nltk import FreqDist
import pandas as pd
import csv

# read preprocessed data, store in lists (clean_data)
reader = open("data_pre_processed.txt", "r")
labels = list()
clean_data = list()
for line in reader.readlines():
    line_split = line.split("\t")
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

# list of words only (excluding pos-tags)
pure_words = [[[word[0:word.index("/")] for word in sent] for sent in text] for text in clean_data]

# list of words only (excluding pos-tags) on text-level
pure_words_text = list()
for text in pure_words:
    temp = list()
    for sent in text:
        for word in sent:
            temp.append(word)
    pure_words_text.append(temp)

# extract 5.000 most frequent words
for doc in pure_words_text:
    freq_dist = FreqDist(doc)
    word_to_num = dict() # maps each of the 5.000 most frequent words to an integer
    count = 1  # zero reserved for words with lower frequency, indicating vocabulary size
    for (word, freq) in freq_dist.most_common(5000):
        word_to_num[word] = count
        count += 1

document_list = list() # contains documents represented as lists of integers
for doc in pure_words_text:
    document = list() # list of integers, non-zero values only
    for word in doc:
        if not (word_to_num.get(word) == None):
            document.append(word_to_num.get(word))
    document_list.append(document)

# get size of longest document
longest_doc = max([len(doc) for doc in pure_words_text])

header = list()
for i in range(1, 501):
    header.append(str("x" + str(i)))
csvfile = open("_data_vectors/index_vector03.csv", "w")
#csvfile = open("_data_vectors_wider/index_vector.csv", "w")
writer = csv.writer(csvfile)
writer.writerow(header)

doc_count = 0
for document in document_list:
    index = 0
    line = ""
    line += str(labels[doc_count]) + ' '
    while index < 500:
        while index < len(document) and index < 500:
            line += str(document[index])+' '
            index += 1
        while index >= len(document) and index < 500:
            line += '0 '
            index += 1
    writer.writerow(line.split())
    doc_count += 1
csvfile.close()