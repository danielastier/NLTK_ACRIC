###import statements
import nltk
from collections import defaultdict
import operator
from collections import OrderedDict
from operator import itemgetter
from nltk.stem.lancaster import LancasterStemmer
from collections import Counter
import pandas as pd


###function_words:
    ##input: string with all words
    ##output: list with dictionary of all function words and their respective number of occurences, total number of function words and percentual relation of function words to the total number of words
def function_words(reader):

    numwords = 0
    allwords = []
    for sent in reader:
        for word in sent:
            word = word.split('/')
            #stem = lancaster_stemmer.stem(word[0])
            #allstems.append(stem)
            numwords = numwords + 1
            allwords.append(word[0])
            
        
    #initialize result list
    resultlist = []
    #initialize dictionary for the function words and their respective occurences
    mydict = {}
    #define a variable for the number of function words
    numfwords = 0
    #define a variable for the percentual relation between the total number of words and the number of function words in the text
    percent_fwords = 0
    #open the list of function words
    with open('functionwords.txt', 'r', encoding='utf-8') as fwords:
        #read in the list of function words and split it (the function words are seperated through commas)
        fwords1 = fwords.read()
        fwordslist = fwords1.split(',')
        
    #iterate the list of function words
    for item in fwordslist:
        #iterate the list of strings    
        for item2 in allwords:
            #compare the respective items to each other
            if item == item2:
                numfwords = numfwords + 1

    if numwords > 0:
        percent_fwords = (numfwords*100)/numwords
    #return resultlist            
    return percent_fwords



    ###adjectives
    ###input: pos_tagged_list
    ##output: list with dictionary of all function words and their respective number of occurences, total number of function words and percentual relation of function words to the total number of words
def adjectives(reader):

    #initialize list
    postaggedlist1 = []
    endlist = ''
    allwords = []
    sentence = ''
    for sent in reader:
        for word in sent:
            word = word.split('/')
            word = tuple(word)
            if len(word) == 2:
                allwords.append(word)
            sentence = sentence + ' ' + word[0]
    numwords = len(allwords)

    #initialize lists for every adjective and adverb form
    normal = []
    comparative = []
    superlative = []
    #initialize dictionaries for all adjectives and adverbs
    alladjs = []
    
    #initialize Boolean variables 
    has_normal = False
    has_comp = False
    has_sup = False

    #define variables for length values
    occurences = 0
    numnormal = 0
    numcomp = 0
    numsup = 0

    #define variables for percentage values
    percent_normal = 0
    percent_comp = 0
    percent_sup = 0


        
##        
    for (word, tag) in allwords:
        
       # print(words)
       # print(tag)
            #continue to use adjectives/adverbs with normal form
        if tag == 'JJ' or tag == 'RB':
            has_normal = True
            numnormal = 1 + numnormal
            
#                    numnormal = numnormal + 1
##                    if words in normaladjs:
##                        normaladjs[words] = normaladjs.get(words) + 1
##                    else:
##                        normaladjs[words] = 1
##                    normaldict = sorted(normaladjs.items(), key = itemgetter(1), reverse = True)
##                    numnormal = numnormal + 1
            
        #continue to use adjectives/adverbs with comparative form
        if tag == 'JJR' or tag == 'RBR':
            numcomp = numcomp + 1
            has_comp = True


            
        #continue to use adjectives/adverbs with superlative form
        if tag == 'JJS' or tag == 'RBS':
            numsup = numsup + 1
            has_sup = True


        
    #define the dictionaries if there is no entry in the list with the respective form
    if not has_normal or numwords < 1:
        percent_normal = 0
    else:
        percent_normal = (numnormal*100)/numwords
    #alladjs.append("percentage of normal adjectives: " + str(percent_normal))
    alladjs.append(percent_normal)
    if not has_comp or numwords < 1:
        percent_comp = 0
    else:
        percent_comp = (numcomp*100)/numwords
    #alladjs.append("percentage of comperative adjectives: " + str(percent_comp))
    alladjs.append(percent_comp)
    if not has_sup or numwords < 1:
        percent_sup = 0
    else:
        percent_sup = (numsup*100)/numwords
    #alladjs.append("percentage of superlative adjectives: " + str(percent_sup))
    alladjs.append(percent_sup)

    #define the total number of adjectives and adverbs
    numtotal = numnormal + numcomp + numsup

    #avoid 'Division by Zero' issue if no adjectives or adverbs are found
    if numtotal == 0:
        alladjs.append(0)
    #define the percentual relation of all adjectives and adverbs to all words and append it to the result list
    else:
        alladjs.append((numtotal*100)/numwords)

    
    
    #return result list which contains a result list for each form
    return alladjs

###det_n:
##input: pos-tagged and tokenized list of input texts
##output: result list with dictionary {Det N bigram:number of occurences}, number of Det N bigrams in general and percentual relation of number of Det N bigrams to number of words in total
def det_n(reader):

    
    allwords = []

    #initialize list of results
    detnlist = []
    #initialize dictionary
    mydict = {}
    sorteddict = {}
    #define boolean variable 'is_DT' which gets the value 'True' if a determiner is followed by a noun and the bigram is a Det N
    is_DT = False
    sentence = ''
    for sent in reader:
        for word in sent:
            word = word.split('/')
            word = tuple(word)
            if len(word) == 2:
                allwords.append(word)
            sentence = sentence + ' ' + word[0]
    numwords = len(allwords)

    #define 'numdetn' for the total number of Det N bigrams in the text
    numdetn = 0
    #-->#remove later --> build class variable to save the total number of words/use length of list
    has_DT_N = False

    #define variable for the percentual relation from Det N bigrams to number of words in general
    percent_Det_N = 0

    #iterate the pos-tagged list of words
    for word, tag in allwords:
        #if the last item of the list was a determiner
        if is_DT:
            #if the current item is a noun
            if 'NN' in tag:
                #verify that the text contains Det N bigrams
                has_DT_N = True
                #continue using the current item
                numdetn = numdetn + 1
                is_DT = False
                

                    
        #use word if it is a determiner            
        if tag == 'DT':
            #set the boolean = True to send the next iteration into the first if-statement:
            is_DT = True
    #define dictionary if no Det N bigrams are in the text
    if has_DT_N == False:
        percent_Det_N = 0

    elif numwords > 0:
        percent_Det_N = (numdetn*100)/numwords
    else:
        percent_Det_N = 0
    #append results to resultlist

    #detnlist.append('percent_Det_N:' + str(percent_Det_N))
    detnlist.append(percent_Det_N)
    #return resultlist
    return detnlist



   ###chunk
   ##input: POS-tagged list
   ##output: number of named entities (geographical positions, persons, locations, organizations)
def chunk_named_entities(reader):

    tagged_sents = list()
    for sent in reader:
        sentence = list()
        for word in sent:
            word = word.split('/')
            word_pos = tuple(word)
            if len(word_pos[0]) > 1:
                sentence.append(word_pos)
        tagged_sents.append(sentence)

    num_words = sum([len([word for word in sent]) for sent in tagged_sents])

    ne_dict = defaultdict(list)
    num_per, num_orga, num_loc, num_gp, num_total = 0, 0, 0, 0, 0
    for sent in tagged_sents:
        ne_sent = nltk.ne_chunk(sent, binary=False)

        for subtree in ne_sent.subtrees(filter=lambda t: t.label() == 'PERSON'):
            num_per += 1
            num_total += 1
        for subtree in ne_sent.subtrees(filter=lambda t: t.label() == 'ORGANIZATION'):
            num_orga += 1
            num_total += 1
        for subtree in ne_sent.subtrees(filter=lambda t: t.label() == 'LOCATION'):
            num_loc += 1
            num_total += 1
        for subtree in ne_sent.subtrees(filter=lambda t: t.label() == 'GPE'):
            num_gp += 1
            num_total += 1

    ne_dict[num_words].append(num_per)
    ne_dict[num_words].append(num_orga)
    ne_dict[num_words].append(num_loc)
    ne_dict[num_words].append(num_gp)
    ne_dict[num_words].append(num_total)

    ne_dist = list()
    for key, value in ne_dict.items():
        if value[0] == 0: # num_per
            ne_dist.append(0)
        else:
            ne_dist.append(value[0]/key)
        if value[1] == 0: # num_orga
            ne_dist.append(0)
        else:
            ne_dist.append(value[1]/key)
        if value[2] == 0: # num_loc
            ne_dist.append(0)
        else:
            ne_dist.append(value[2]/key)
        if value[3] == 0: # num_gp
            ne_dist.append(0)
        else:
            ne_dist.append(value[3]/key)
        if value[4] == 0: # num_total
            ne_dist.append(0)
        else:
            ne_dist.append(value[4]/key)

    return ne_dist


reader = open("data_pre_processed.txt", 'r', encoding = 'utf-8')
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

header = list()
func_words = list()
header.append("funct_words")
det_noun = list()
header.append("det_n")
norm = list()
header.append("norms")
comp = list()
header.append("comps")
superl = list()
header.append("supers")
alladj = list()
header.append("all_adj")
ne_geo_pos = list()
header.append("ne_geo_pos")
ne_pers = list()
header.append("ne_persons")
ne_loc = list()
header.append("ne_locations")
ne_org = list()
header.append("ne_organizations")
ne_total = list()
header.append("ne_total")

for entry in clean_data:
    func_words.append(function_words(entry))
    det_noun.append(det_n(entry))
    adj = adjectives(entry)
    norm.append(adj[0])
    comp.append(adj[1])
    superl.append(adj[2])
    alladj.append(adj[3])
    ne_chunks = chunk_named_entities(entry)
    ne_pers.append(ne_chunks[0])
    ne_org.append(ne_chunks[1])
    ne_loc.append(ne_chunks[2])
    ne_geo_pos.append(ne_chunks[3])
    ne_total.append(ne_chunks[4])

# store vectorized data in 'test_data_sent_vector.csv' file, including labels at first position
#data_matrix = pd.concat([pd.DataFrame(labels), pd.DataFrame(func_words), pd.DataFrame(det_noun), pd.DataFrame(norm), pd.DataFrame(comp), pd.DataFrame(superl), pd.DataFrame(alladj)], axis=1)
data_matrix = pd.concat([pd.DataFrame(func_words), pd.DataFrame(det_noun), pd.DataFrame(norm), pd.DataFrame(comp), pd.DataFrame(superl), pd.DataFrame(alladj), pd.DataFrame(ne_pers), pd.DataFrame(ne_org), pd.DataFrame(ne_loc), pd.DataFrame(ne_geo_pos), pd.DataFrame(ne_total)], axis=1)
data_matrix.to_csv('_data_vectors/sent02_vector.csv', index=False, delimiter=',', header=header)
#data_matrix.to_csv('_data_vectors_wider/sent02_vector.csv', index=False, delimiter=',', header=header)

