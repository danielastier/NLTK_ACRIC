__author__ = 'Daniela Stier'

### IMPORT STATEMENTS
import string
import pandas as pd
import re as re
from collections import defaultdict, OrderedDict, Counter

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

# clean data excluding punctuation marks
puncts = string.punctuation
clean_data = [[[token  for token in sent if not token[0:token.index("/")].lower() in puncts] for sent in text] for text in clean_data]

# list of words only (excluding pos-tags)
pure_words = [[[word[0:word.index("/")] for word in sent] for sent in text] for text in clean_data]
# # list of words only excluding stopwords (excluding pos-tags)
# pure_words_wo_stops = [[[word[0:word.index("/")] for word in sent] for sent in text] for text in clean_data_wo_stops]
# list of words only (excluding pos-tags) on text-level
pure_words_text = list()
for text in pure_words:
    temp = ""
    for sent in text:
        for word in sent:
            temp += word + " "
    pure_words_text.append(temp)


######### MEASURES CONCERNING CHARACTERS ##############################

header = list()
# number of characters
num_chars = [[[len(word) for word in sent] for sent in text] for text in pure_words]
num_chars_summed_sent = [[sum(sent) for sent in text] for text in num_chars]
num_chars_summed = [sum(text) for text in num_chars_summed_sent]
header.append("num_char")
# # number of characters excluding stopwords
# num_chars_wo_stops = [[[len(word) for word in sent] for sent in text] for text in pure_words_wo_stops]
# num_chars_wo_stops_summed_sent = [[sum(sent) for sent in text] for text in num_chars_wo_stops]
# num_chars_wo_stops_summed = [sum(text) for text in num_chars_wo_stops_summed_sent]

# frequency of characters
# rel. freq of each char: dividing the frequency of that char in a text by the total frequencies of chars
letters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzĀāġĪīŠšŪūǧʿḍḏḡḥḫḵṣṭṯẓ‘"
counter = defaultdict(list)
for i in letters:
    header.append(str("char_freq_dist") + str(i))
    for text in pure_words_text:
        temp = Counter(text)
        counter[i].append(temp[i])
counter = OrderedDict(sorted(counter.items(), key=lambda x: x[0]))
char_freq_dist = list()
for value in counter.values():
    temp = list()
    for i in range(len(num_chars_summed)):
        if num_chars_summed[i] > 0:
            temp.append(value[i]/num_chars_summed[i])
        else:
            temp.append(0)
    char_freq_dist.append(temp)

# store vectorized data in 'test_data_char_vector.csv' file, including labels at first position
#data_matrix = pd.concat([pd.DataFrame(labels), pd.DataFrame(num_chars_summed), pd.DataFrame(char_freq_dist).transpose()], axis=1)
data_matrix = pd.concat([pd.DataFrame(num_chars_summed), pd.DataFrame(char_freq_dist).transpose()], axis=1)
data_matrix.to_csv('_data_vectors/char_vector.csv', index=False, delimiter=',', header=header)

# # character frequencies (in- and excluding stopwords)
# charFreq_stops_list1 = chars.char_frequency(text1_wordList)
# charFreq_noStops_list1 = chars.char_frequency(text1_wordNoStopsList)