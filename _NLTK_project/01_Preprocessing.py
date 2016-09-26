__author__ = 'Daniela Stier'

### IMPORT STATEMENTS
import re
import os as os
from nltk.tokenize import TweetTokenizer, sent_tokenize
from nltk import pos_tag, bigrams, corpus
from bs4 import BeautifulSoup


############### IMPLEMENTATION OF HELPER METHODS ###############
### read in input text
def read_file(input_file):
    reader = open(input_file, 'r', encoding='utf-8')#encoding='ISO-8859-1')
    intext = reader.read()
    reader.close()
    # exclude HTML tags from data
    intext_clean = BeautifulSoup(intext, "html.parser")
    intext_clean = intext_clean.get_text()
    return intext_clean

### normalize content
def normalize(input_text):
    input_clean = re.sub('[“”]', '"', input_text)
    input_clean = re.sub('[‘’]', "'", input_clean)
    #input_clean = re.sub(r'\."\s[(.*)]', r'"\s[(\.*)].', input_clean)
    return input_clean

### return individual sentences of an input text
def extract_sentences(input_text):
    sentences = sent_tokenize(input_text)
    return sentences

### tokenize input text - nltk.TweetTokenizer does not split contractions
def tokenize_sentences(input_text):
    sent_tokens = [TweetTokenizer().tokenize(sent) for sent in input_text]
    return sent_tokens

### add pos-information to sentences of an input text - nltk as reliable
def pos_tag_sentences(input_text):
    sent_tagged = [pos_tag(sent) for sent in input_text]
    return sent_tagged

### change pos-notation
def change_pos_notation(input_text):
    sent_tagged = [[str(word + "/" + tag) for (word, tag) in sent] for sent in input_text]
    return sent_tagged





############### PREPROCESSING ###############
#path_normal = "../_NLTK_corpus/_merged_DS_test/non_radical/"
#path_radical = "../_NLTK_corpus/_merged_DS_test/radical/"
path_normal = "../_NLTK_corpus/_NLTK_corpus/non_radical/"
path_radical = "../_NLTK_corpus/_NLTK_corpus/radical/"
normal = "nor"
radical = "rad"

class_labels = ((path_normal, normal), (path_radical, radical))

# read in data, pre-process input texts, create bigrams and unigram
# output: all sentences of one document to a single line, including class labels
with open("data_pre_processed.txt", 'a', newline='') as w:
    for entry in class_labels:
        label = entry[1]
        # read all documents in directory
        for file in os.listdir(entry[0]):
            if not file == ".DS_Store":
                w.write(label + "\t")
                # read file content
                content = read_file(str(entry[0] + file))
                # normalize content to enable exact sentence segmentation
                cont_post = normalize(content)
                # extract sentences
                cont_sentences = extract_sentences(cont_post)
                # tokenize sentences
                cont_tokens = tokenize_sentences(cont_sentences)
                # pos-tag sentences
                cont_tags = pos_tag_sentences(cont_tokens)
                # rewrite pos-tag information
                cont_tags = change_pos_notation(cont_tags)
                # store pre-processed data in 'data_pre_processed.txt' file
                for sent in cont_tags:
                    if not len(sent) == 0:
                        w.write(str(sent) + "\t")
            w.write("\n")
        print(label)
    w.close()
