__author__ = 'Daniela Stier'

### IMPORT STATEMENTS
import string
import pandas as pd
from collections import defaultdict, OrderedDict, Counter
from nltk import corpus

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

# list of words only (excluding pos-tags)
pure_words = [[[token[0:token.index("/")] for token in sent if not token[0:token.index("/")].lower() in puncts] for sent in text] for text in clean_data]
# list of words only (excluding pos-tags) on text-level
pure_words_text = list()
for text in pure_words:
    temp = ""
    for sent in text:
        for word in sent:
            temp += word + " "
    pure_words_text.append(temp)

# list of puncts only
pure_puncts = [[[punct[0:punct.index("/")] for punct in sent if punct[0:punct.index("/")] in puncts] for sent in text] for text in clean_data]
# list of puncts only on text level
pure_puncts_text = list()
for text in pure_puncts:
    temp = ""
    for sent in text:
        for punct in sent:
            temp += punct + " "
    pure_puncts_text.append(temp)

# number of words
num_words = [[len(sent) for sent in text] for text in pure_words]
num_words_summed = [sum(sent) for sent in num_words]

# number of characters
num_chars = [[[len(word) for word in sent] for sent in text] for text in pure_words]
num_chars_summed_sent = [[sum(sent) for sent in text] for text in num_chars]
num_chars_summed = [sum(text) for text in num_chars_summed_sent]

######### MEASURES CONCERNING PUNCTUATION MARKS ##############################

header = list()
# number of punctuation marks
num_puncts = [[len(sent) for sent in text] for text in pure_puncts]
num_puncts_summed = [sum(sent) for sent in num_puncts]
header.append("num_punct")

# frequency of punctuation marks
# rel. freq of each punct: dividing the frequency of that punct in a text by the total frequencies of puncts
counter = defaultdict(list)
header_temp_puncts = list()
header_temp_chars = list()
header_temp_words = list()
c = 1
for i in puncts:
    header_temp_puncts.append(str("punct_freq_dist_puncts") + str(c))
    header_temp_chars.append(str("punct_freq_dist_chars") + str(c))
    header_temp_words.append(str("punct_freq_dist_words") + str(c))
    print(str(c) + ": " + str(i))
    c += 1
    for text in pure_puncts_text:
        temp = Counter(text)
        counter[i].append(temp[i])
counter = OrderedDict(sorted(counter.items(), key=lambda x: x[0]))
punct_freq_dist_puncts = list()
punct_freq_dist_words = list()
punct_freq_dist_chars = list()
for value in counter.values():
    temp = list()
    for i in range(len(num_puncts_summed)):
        if num_puncts_summed[i] == 0:
            temp.append(0)
        elif not num_puncts_summed[i] == 0:
            temp.append(value[i]/num_puncts_summed[i])
    punct_freq_dist_puncts.append(temp)
    temp = list()
    for i in range(len(num_chars_summed)):
        if num_chars_summed[i] == 0:
            temp.append(0)
        elif not num_chars_summed[i] == 0:
            temp.append(value[i]/num_chars_summed[i])
    punct_freq_dist_chars.append(temp)
    temp = list()
    for i in range(len(num_words_summed)):
        if num_words_summed[i] == 0:
            temp.append(0)
        elif not num_words_summed[i] == 0:
            temp.append(value[i]/num_words_summed[i])
    punct_freq_dist_words.append(temp)

for i in header_temp_puncts:
    header.append(i)
for i in header_temp_chars:
    header.append(i)
for i in header_temp_words:
    header.append(i)
# store vectorized data in 'test_data_puncts_vector.csv' file, including labels at first position
#data_matrix = pd.concat([pd.DataFrame(labels), pd.DataFrame(num_puncts_summed), pd.DataFrame(punct_freq_dist_puncts).transpose(), pd.DataFrame(punct_freq_dist_chars).transpose(), pd.DataFrame(punct_freq_dist_words).transpose()], axis=1)
data_matrix = pd.concat([pd.DataFrame(num_puncts_summed), pd.DataFrame(punct_freq_dist_puncts).transpose(), pd.DataFrame(punct_freq_dist_chars).transpose(), pd.DataFrame(punct_freq_dist_words).transpose()], axis=1)
#data_matrix.to_csv('_data_vectors/puncts_vector.csv', index=False, delimiter=',', header=header)


# # sentence mark frequencies
# sentFreq_stops_list1 = puncts.sent_frequency(text1_punctList)
# av_sentFreq_stops_text1 = calcs.mean_lemma_pos_freq(sentFreq_stops_list1)