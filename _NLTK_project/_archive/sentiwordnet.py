from nltk.corpus import sentiwordnet as swn
import re
from nltk.compat import python_2_unicode_compatible
from nltk.corpus.reader import CorpusReader
import nltk
from nltk.corpus import wordnet
import nltk.data
from nltk.stem import WordNetLemmatizer

def posneg(textfile):
    wordnet_lemmatizer = WordNetLemmatizer()
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_detector.tokenize(textfile.strip())
    numsentences = len(sentences)
    allpos = 0
    allneg = 0
    allsyns = 0
    possent = 0
    negsent = 0
    nnew = list()
    sentlist = []
    allsents = []
    #print('numsentences')
    #print(numsentences)
    for sentence in sentences:
        
        #print(sentence)
        
        sentlist.append(sentence)
        words = nltk.word_tokenize(sentence)
        words = nltk.pos_tag(words)
        for word, tag in words:
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
##                            if posscore + negscore > 0:
##                                pass
##                                print('item')
##                                print(itemlesk)
##                                print('posscore')
##                                print(posscore)
##                                print('negscore')
##                                print(negscore)
                            
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
##                            if posscore + negscore > 0:
##                                
##                                print('item')
##                                print(itemlesk)
##                                print('posscore')
##                                print(posscore)
##                                print('negscore')
##                                print(negscore)
                            
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
##                            if posscore + negscore > 0:
##                                
##                                print('item')
##                                print(itemlesk)
##                                print('posscore')
##                                print(posscore)
##                                print('negscore')
##                                print(negscore)
                            
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
            
##                            if posscore + negscore > 0:
##                             
##                                print('item')
##                                print(itemlesk)
##                                print('posscore')
##                                print(posscore)
##                                print('negscore')
##                                print(negscore)
##        print('possent:')
##        print(possent)
##        print('negsent:')
##        print(negsent)
##        if possent > negsent:
##            print('sent is positive!')
##            possent = negsent = 0
##        if possent < negsent:
##            print('sent is negative!')
##            possent = negsent = 0
##        else:
##            print('neutral!')
##            possent = negsent = 0
            
        
            
        
        
##    print('allpos')
##    print(allpos)
##    print('allneg')
##    print(allneg)
##    print('allsyns')
##    print(allsyns)
    pos = allpos/allsyns
    neg = allneg/allsyns
    overall = pos - neg
                            
##    print('allpos/allsyns')
##    print(allpos/allsyns)
##    print('allneg/allsyns')
##    print(allneg/allsyns)
    return(overall)
    
            

                
