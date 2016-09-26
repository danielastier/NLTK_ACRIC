__author__ = 'Lisa Hiller'

# import statements
import textrazor
import nltk
from collections import OrderedDict
from operator import itemgetter
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import sentiwordnet as swn
import re
from nltk.compat import python_2_unicode_compatible
from nltk.corpus.reader import CorpusReader
import nltk
from nltk.corpus import wordnet
import nltk.data
from nltk.stem import WordNetLemmatizer
import pandas as pd

reader = open("data_pre_processed.txt", 'r', encoding = 'utf-8')

#class Content(object):
        
    ###vocabulary richness:
    
    ### ---> list with ALL tokenized words.

def num_words(listsent):
    allwords = 0

    for sent in listsent:
        for word in sent:
            allwords = allwords + 1
    return allwords

#numwords = numwords(reader)

def vocabulary_richness(reader):

    allstems = []
    lancaster_stemmer = LancasterStemmer()
    numwords = 0
    for sent in reader:
        for word in sent:
            word = word.split('/')
            stem = lancaster_stemmer.stem(word[0])
            allstems.append(stem)
            numwords = numwords + 1
    allstems = set(allstems)
    #print(allstems)
    #print('DONE!')
    numstems = len(allstems)
    if numwords > 1:
        ratio = numstems / numwords
    else:
        ratio = 0
    return ratio
            
            
            
            

###unique_features:
##input: complete txt file
##compares input file to list of typical islamistic expressions (unique features) and words
##returns list with overall results and detailled results
##-->complete text string
def higher_needs(reader):
    allwords = []
    existing_features = 0
    percent_unique_features = 0
    numwords = num_words(reader)
    for sent in reader:
        for word in sent:
            word = word.split('/')
            allwords.append(word[0])
            
    #read in list of higher level needs and save it to a variable 'features'
    with open ('dict_needs.txt', 'r', encoding='utf-8') as keywords:
        lines = keywords.read()
        hasfeature = False
        
        #iterate the list of unique features
        for item in lines.split('\n'):
            for word in allwords:
                if item == word:
                    existing_features = existing_features + 1
                    hasfeature = True
                    
        if hasfeature:
            percent_unique_features = existing_features*100/numwords
        else:
            percent_unique_features = 0

    return percent_unique_features

def lower_needs(reader):

    allwords = []
    existing_features = 0
    percent_unique_features = 0
    numwords = num_words(reader)
    for sent in reader:
        for word in sent:
            word = word.split('/')
            allwords.append(word[0])
            
    #read in list of higher level needs and save it to a variable 'features'
    with open ('dict_lower.txt', 'r', encoding='utf-8') as keywords:
        lines = keywords.read()
        hasfeature = False
        
        #iterate the list of unique features
        for item in lines.split('\n'):
            for word in allwords:
                if item == word:
                    existing_features = existing_features + 1
                    hasfeature = True
                    
        if hasfeature:
            percent_unique_features = existing_features*100/numwords
        else:
            percent_unique_features = 0

        return percent_unique_features

def dict_radicalness(reader):

    allwords = []
    instring = ''
    existing_features = 0
    percent_unique_features = 0
    numwords = num_words(reader)
    for sent in reader:
        for word in sent:
            word = word.split('/')
            instring = instring + ' ' + word[0].lower()
            

    #read in list of higher level needs and save it to a variable 'features'
    with open ('dict_radicalism.txt', 'r', encoding='utf-8') as keywords:
        lines = keywords.read()
        lines = lines.lower()
        hasfeature = False

        #iterate the list of unique features
        #for word in allwords:
        for item in lines.split('\n'):
    
            if item.lower() in instring:
                #print(item)
                numocs = instring.count(item)
                existing_features = existing_features + numocs
                hasfeature = True

        if hasfeature:
            percent_unique_features = existing_features*100/numwords
        else:
            percent_unique_features = 0

        return percent_unique_features

def jihad_verses(reader):
    resultlist = list()
    ratio = 0
    instring = ""
    for sent in reader:
        for word in sent:
            word = word.split('/')
            instring = instring + word[0].lower()
            
    verse_list = re.findall(r'\[([^]]*)\]',instring)
    #verse_list = verse_list.replaceall(' ', '')
    numverses = len(verse_list)
    numjihadverses = 0
    #print("verse_list")
    #print(verse_list)
    with open('jihad_verses.txt', 'r', encoding = 'utf-8') as verses:
        verses = verses.read()
        verses = verses.splitlines()
        for item in verse_list:
            item = item.replace(": ", ":")
            for reference in verses:
                #print(reference)
                reference = reference.split()
                if len(reference) > 1:
                    if reference[0] == item or reference[1].lower() == item:
                        resultlist.append(item)

                        #print(resultlist)
        numjihadverses = len(resultlist)
        #print(numjihadverses)
                
    if numverses > 0:
        ratio = numjihadverses/numverses
    return ratio
    
                    
                    
###topic_tagger:
##input: complete txt file/string
##uses the TextRazor tool to identify the most probable topic of the text
#---->#use the complete txt-file/ string
##def topic_tagger(string):
##    header.append("topics")
##    endlist = ''
##    allwords = ''
##    with open(string, 'r', encoding = 'utf-8') as myfile:
##        myfile = myfile.readlines()
##        for item in myfile:
####                item = item.replace('	', '')
####                item = item.replace('[', ',')
####                item = item.replace(']', '')
####                item = item.replace(',,', ', ')
####                item = item.replace('.', '')
####                item = item.replace('?', '')
####                item = item.replace('!', '')
####                item = item.replace(';', '')
####                item = item.replace(':', '')
##            item = item.split(',')
##            
##            
##            for items in item:
##                #print(items)
##                words = items.split('/')
##                words = words[0]
##                words = words[2:]
##                allwords = allwords + ' ' + words
##                allwords = allwords.replace('    ', ' ')
##            string = allwords
##            #print(string)
##    #define the api key to use textrazor
##    textrazor.api_key = 'a97ff131bf20488f686cda471f70586e7c9c8922b6eea25148d2a9fa'
##    #define the extractors "entities" and "topics"
##    client = textrazor.TextRazor(extractors=["entities", "topics"])
##    #analyze the string regarding its entities and topics
##    response = client.analyze(string)
##    #create a list with all found entities
##    entities = list(response.entities())
##    #sort the entities' list regarding their relevance score. The higher the relevance score, the higher the probability that a text is about the mentioned topic
##    entities.sort(key=lambda x: x.relevance_score, reverse=True)
##    #initialize a dictionary that contains the respective topic and their respective relevance score
##    mydict = {}
##    #iterate the list of entities
##    for entity in entities:
##        #define the relevance score
##        rel = entity.relevance_score
##        #add the entities to the dictionary if they are not inside yet
##        if entity.id not in mydict:
##            mydict[entity.id] = rel
##            #convert the relevance score to string format
##            rel = str(rel)
##            #sort the dictionary according to its relevance score reversely and save it to a variable 'sorteddict'
##            sorteddict = OrderedDict(sorted(mydict.items(), key =itemgetter(1), reverse=True))
##    #return dictionary
##    return sorteddict


def posneg(reader):

    allwords = []
    alltags = []
    sentlist = []
    allsyns = 0
    allpos = 0
    allneg = 0
    possent = 0
    negsent = 0
    wordnet_lemmatizer = WordNetLemmatizer()
    sentstring = ''
    sentence = ''
    #sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    for sent in reader:
        
        for word in sent:
            word = word.split('/')
            word = tuple(word)
            if len(word) == 2:
                allwords.append(word)
            sentence = sentence + ' ' + word[0]
            #print(tuple(word))
            
        
            
    
        #print(allwords)
        for word, tag in allwords:
            if tag.startswith('N'):
                word = wordnet_lemmatizer.lemmatize(word, wordnet.NOUN)
                
                allsyns = allsyns + 1
                
                nnew = list(swn.senti_synsets(word, 'n'))

                
                itemlesk = nltk.wsd.lesk(sentence, word, 'n')
                
                if itemlesk is not None:
                    itemlesk = itemlesk.name()

                    for item in nnew:
                        itemname = item.synset.name()

                        if itemlesk == itemname:

                            synset = swn.senti_synset(itemname)
                            posscore = synset.pos_score()
                            negscore = synset.neg_score()
                            sentlist.append((itemname, posscore, negscore))
                            allpos = allpos + posscore
                            allneg = allneg + negscore
                            possent = posscore + possent
                            negsent = negscore + negsent

                            
            if tag.startswith('J'):
                allsyns = allsyns + 1
                word = wordnet_lemmatizer.lemmatize(word, wordnet.ADJ)
                nnew = list(swn.senti_synsets(word, 'a'))

                
                itemlesk = nltk.wsd.lesk(sentence, word, 'a')
                
                if itemlesk is not None:
                    itemlesk = itemlesk.name()

                    for item in nnew:
                        itemname = item.synset.name()

                        if itemlesk == itemname:
                            

                            synset = swn.senti_synset(itemname)
                            posscore = synset.pos_score()
                            negscore = synset.neg_score()
                            allpos = allpos + posscore
                            allneg = allneg + negscore
                            possent = posscore + possent
                            negsent = negscore + negsent

                            
            if tag.startswith('V'):
                allsyns = allsyns + 1
                word = wordnet_lemmatizer.lemmatize(word, wordnet.VERB)
                nnew = list(swn.senti_synsets(word, 'v'))

                
                itemlesk = nltk.wsd.lesk(sentence, word, 'v')
                
                if itemlesk is not None:
                    itemlesk = itemlesk.name()

                    for item in nnew:
                        itemname = item.synset.name()

                        if itemlesk == itemname:
                            

                            synset = swn.senti_synset(itemname)
                            posscore = synset.pos_score()
                            negscore = synset.neg_score()
                            allpos = allpos + posscore
                            allneg = allneg + negscore
                            possent = posscore + possent
                            negsent = negscore + negsent

                            
            if tag.startswith('R'):
                allsyns = allsyns + 1
                word = wordnet_lemmatizer.lemmatize(word, wordnet.ADV)
                nnew = list(swn.senti_synsets(word, 'r'))

                
                itemlesk = nltk.wsd.lesk(sentence, word, 'r')
                
                if itemlesk is not None:
                    itemlesk = itemlesk.name()

                    for item in nnew:
                        itemname = item.synset.name()

                        if itemlesk == itemname:

                            synset = swn.senti_synset(itemname)
                            posscore = synset.pos_score()
                            negscore = synset.neg_score()
                            allpos = allpos + posscore
                            allneg = allneg + negscore
                            possent = posscore + possent
                            negsent = negscore + negsent
            sentence = ''
            allwords = []
    if allsyns > 0:
        pos = allpos/allsyns
        neg = allneg/allsyns
        overall = pos - neg
    else:
        overall = 0

    return(overall)

# def all_results(reader):
#     resultlist = list()
#     voc_rich = vocabulary_richness(reader)
#     high_needs = higher_needs(reader)
#     low_needs = lower_needs(reader)
#     senti_score = posneg(reader)
#     #to be done:
#     #topic_tagger = 0
#     resultlist.append('vocabulary richness:' + str(voc_rich))
#     resultlist.append('higher level needs:' + str(high_needs))
#     resultlist.append('lower level needs:' + str(low_needs))
#     resultlist.append('positive/negative sentiment:' + str(senti_score))
#     #resultlist.append('relevant topics:' + str(topic_tagger))
#    return resultlist

reader = open("data_pre_processed.txt", 'r', encoding = 'utf-8')
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
voc_rich = list()
header.append("voc_richness")
high_needs = list()
header.append("higher_needs")
low_needs = list()
header.append("lower_needs")
radicality = list()
header.append("rad_dict")
senti_score = list()
header.append("sentiwordnet")
jihad_list = list()
header.append("jihadlist")

for entry in clean_data:
    voc_rich.append(vocabulary_richness(entry))
    high_needs.append(higher_needs(entry))
    low_needs.append(lower_needs(entry))
    radicality.append(dict_radicalness(entry))
    senti_score.append(posneg(entry))
    jihad_list.append(jihad_verses(entry))
    

# store vectorized data in 'test_data_sent_vector.csv' file, including labels at first position
#data_matrix = pd.concat([pd.DataFrame(labels), pd.DataFrame(voc_rich), pd.DataFrame(high_needs), pd.DataFrame(low_needs), pd.DataFrame(senti_score)], axis=1)
data_matrix = pd.concat([pd.DataFrame(voc_rich), pd.DataFrame(high_needs), pd.DataFrame(low_needs), pd.DataFrame(radicality), pd.DataFrame(senti_score), pd.DataFrame(jihad_list)], axis=1)
data_matrix.to_csv('_data_vectors/sent01_vector.csv', index=False, delimiter=',', header=header)
