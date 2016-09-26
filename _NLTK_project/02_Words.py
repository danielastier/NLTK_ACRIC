__author__ = 'Daniela Stier'

### IMPORT STATEMENTS
import string
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
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
clean_data = [[[token for token in sent if not token[0:token.index("/")].lower() in puncts] for sent in text] for text in clean_data]

# clean_data excluding stopwords
stops = corpus.stopwords.words("english")
clean_data_wo_stops = [[[token for token in sent if not token[0:token.index("/")].lower() in stops] for sent in text] for text in clean_data]

# list of words only (excluding pos-tags)
pure_words = [[[word[0:word.index("/")] for word in sent] for sent in text] for text in clean_data]
# list of words only (excluding pos-tags) on text-level
pure_words_text = list()
for text in pure_words:
    temp = ""
    for sent in text:
        for word in sent:
            temp += word + " "
    pure_words_text.append(temp)

# list of words only excluding stopwords (excluding pos-tags)
pure_words_wo_stops = [[[word[0:word.index("/")] for word in sent] for sent in text] for text in clean_data_wo_stops]

# list tags only (excluding tokens)
pure_tags_sent = [[[tag[tag.index("/")+1:] for tag in sent] for sent in text] for text in clean_data]
pure_tags_text = list()
for text in pure_tags_sent:
    temp = list()
    for sent in text:
        temp.append(sent)
    pure_tags_text.append(temp)
num_tags = [[len(sent) for sent in text] for text in pure_tags_sent]
num_tags_summed = [sum(sent) for sent in num_tags]


######### MEASURES CONCERNING WORDS ##############################

header = list()
# number of words
num_words = [[len(sent) for sent in text] for text in clean_data]
num_words_summed = [sum(sent) for sent in num_words]
# number of words excluding stopwords
num_words_wo_stops = [[len(sent) for sent in text] for text in clean_data_wo_stops]
num_words_wo_stops_summed = [sum(sent) for sent in num_words_wo_stops]
header.append("num_word")
header.append("num_word_wo_stops")

# number of characters
num_chars = [[[len(word) for word in sent] for sent in text] for text in pure_words]
num_chars_summed_sent = [[sum(sent) for sent in text] for text in num_chars]
num_chars_summed = [sum(text) for text in num_chars_summed_sent]
# number of characters excluding stopwords
num_chars_wo_stops = [[[len(word) for word in sent] for sent in text] for text in pure_words_wo_stops]
num_chars_wo_stops_summed_sent = [[sum(sent) for sent in text] for text in num_chars_wo_stops]
num_chars_wo_stops_summed = [sum(text) for text in num_chars_wo_stops_summed_sent]

# average word length including stop words
# dividing the total number of characters in a text by the total number of words in a text
av_word_length = list()
for i, j in zip(num_chars_summed, num_words_summed):
    if j > 0:
        av_word_length.append(i/j)
    else:
        av_word_length.append(0)
#av_word_length = [num_char/num_word for num_char, num_word in zip(num_chars_summed, num_words_summed)]
header.append("av_word_len")
# average word length excluding stopwords, output: float(number)
av_word_length_wo_stops = list()
for i, j in zip(num_chars_wo_stops_summed, num_words_wo_stops_summed):
    if j > 0:
        av_word_length_wo_stops.append(i/j)
    else:
        av_word_length_wo_stops.append(0)
#av_word_length_wo_stops = [num_char/num_word for num_char, num_word in zip(num_chars_wo_stops_summed, num_words_wo_stops_summed)]
header.append("av_word_len_wo_stops")

# frequency of part-of-speech tags
# rel. freq of each pos tag: dividing the frequency of that tag in a text by the total frequencies of tags
pos_tags = "CC,CD,DT,EX,FW,IN,JJ,JJR,JJS,LS,MD,NN,NNS,NNP,NNPS,PDT,POS,PRP,PRP$,RB,RBR,RBS,RP,SYM,TO,UH,VB,VBD,VBG,VBN,VBP,VBZ,WDT,WP,WP$,WRB"
counter = defaultdict(str)
for i in pos_tags.split(","):
    counter[i] = [text.count(i) for text in pure_tags_text]
    header.append(str("pos_freq_dist") + str(i))
counter = OrderedDict(sorted(counter.items(), key=lambda x: x[0]))
pos_freq_dist = list()
for value in counter.values():
    temp = list()
    for i in range(len(num_tags_summed)):
        if num_tags_summed[i] > 0:
            temp.append(value[i]/num_tags_summed[i])
        else:
            temp.append(0)
    pos_freq_dist.append(temp)

# [1|2|...|30]-character words distribution (according to the whole text, based on characters)
# rel. freq of each word-length: dividing total number of words of that length by total number of words
wos = list()
for text in num_chars:
    temp = list()
    for sent in text:
        for word in sent:
            temp.append(word)
    wos.append(temp)
max_num_char = [np.mean(sent) for sent in wos]
print(max_num_char)
counter = defaultdict(int)
c = 0
for i in range(int(max_num_char[c])-10, int(max_num_char[c])+15):
#for i in range(1, int(max_num_char)+70):
    counter[i] = [text.count(i) for text in num_chars_summed_sent]
    header.append(str("word_len_dist_chars") + str(i))
    c += 1

word_length_dist_chars = list()
for value in counter.values():
    temp = list()
    for i in range(len(num_words_summed)):
        if num_words_summed[i] > 0:
            temp.append(value[i]/num_words_summed[i])
        else:
            temp.append(0)
    word_length_dist_chars.append(temp)

# personal pronoun frequency distribution
# rel. freq of each pronoun: dividing total number of pronoun by total number of words
pers_pron = "he,he'd,he'll,he's,her,hers,herself,him,himself,his"
pers_pron += "i,i'd,i'll,i'm,i've,it,it's,its,itself,let's,me,my,myself"
pers_pron += "our,ours,ourselves,she,she'd,she'll,she's,their,theirs,them,themselves"
pers_pron += "they,they'd,they'll,they're,they've,us,we,we'd,we'll,we're,we've"
pers_pron += "you,you'd,you'll,you're,you've,your,yours,yourself,yourselves"
counter = defaultdict(str)
c = 1
for i in pers_pron.split(","):
    counter[i] = [text.count(i) for text in pure_words_text]
    header.append(str("pers_pron") + str(c))
    c += 1
counter = OrderedDict(sorted(counter.items(), key=lambda x: x[0]))
prp_freq_dist = list()
for value in counter.values():
    temp = list()
    for i in range(len(num_words_summed)):
        if num_words_summed[i] > 0:
            temp.append(value[i]/num_words_summed[i])
        else:
            temp.append(0)
    prp_freq_dist.append(temp)

# tense distribution
# rel. freq of tense uses: dividing total number of tense indicating pos-tag by total number of tense-tags
pos_tense = "VB,VBD,VBG,VBN,VBP,VBZ"
total_tense = [len([tag for tag in sent if tag in pos_tense.split(",")]) for sent in pure_tags_text]
counter = defaultdict(str)
for i in pos_tense.split(","):
    counter[i] = [text.count(i) for text in pure_tags_text]
    header.append(str("tense_") + str(i))
counter = OrderedDict(sorted(counter.items(), key=lambda x: x[0]))
tense_freq_dist = list()
for value in counter.values():
    temp = list()
    for i in range(len(total_tense)):
        if total_tense[i] > 0:
            temp.append(value[i]/total_tense[i])
        else:
            temp.append(0)
    tense_freq_dist.append(temp)


# store vectorized data in 'test_data_word_vector.csv' file, including labels at first position
#data_matrix = pd.concat([pd.DataFrame(labels), pd.DataFrame(num_words_summed), pd.DataFrame(num_words_wo_stops_summed), pd.DataFrame(av_word_length), pd.DataFrame(av_word_length_wo_stops), pd.DataFrame(pos_freq_dist).transpose(), pd.DataFrame(word_length_dist_chars).transpose(), pd.DataFrame(prp_freq_dist).transpose()], pd.DataFrame(tense_freq_dist).transpose()], axis=1)
data_matrix = pd.concat([pd.DataFrame(num_words_summed), pd.DataFrame(num_words_wo_stops_summed), pd.DataFrame(av_word_length), pd.DataFrame(av_word_length_wo_stops), pd.DataFrame(pos_freq_dist).transpose(), pd.DataFrame(word_length_dist_chars).transpose(), pd.DataFrame(prp_freq_dist).transpose(), pd.DataFrame(tense_freq_dist).transpose()], axis=1)
data_matrix.to_csv('_data_vectors/word_vector.csv', index=False, delimiter=',', header=header)





# ### path similarity between sentences occurring in one text
# av_path_similarity_score1 = Path_similarity_lesk.calculate_path_similarity(text1_sentences)
# av_path_similarity_score2 = Path_similarity_lesk.calculate_path_similarity(text2_sentences)
# av_path_similarity_score = av_path_similarity_score1 - av_path_similarity_score2


# ### big word ratio (range from 2 to longest word)
# # dividing total number of words in a text by the total number of words with more than num characters
# print(num_chars)
# longest_words = max([max([max([word for word in sent]) for sent in text]) for text in num_chars])
# print(longest_words)
# counter = defaultdict(int)
# for i in range(2, longest_words+1):
#     for n in range(i, longest_words+1):
#         counter[i] = [text.count(n) for text in num_chars]
# print(counter)