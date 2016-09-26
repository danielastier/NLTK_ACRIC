import re
import string
def jihad_verses(reader):
    resultlist = []
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
    print("verse_list")
    print(verse_list)
    with open('jihad_verses.txt', 'r', encoding = 'utf-8') as verses:
        verses = verses.read()
        verses = verses.splitlines()
        for item in verse_list:
            item = item.replace(": ", ":")
            for reference in verses:
                reference = reference.split()
                print("reference")
                print(reference)
                print("item")
                print(item)
                if len(reference) > 1:
                    if reference[0].lower() == item or reference[1].lower() == item:
                        numjihadverses = numjihadverses + 1
                        print("numjihadverses:")
                        print(numjihadverses)
    if numverses > 0:
        ratio = numjihadverses/numverses

    return ratio
