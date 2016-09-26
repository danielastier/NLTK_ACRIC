import re

def jihad_verses(textfile):
    resultlist = []
    verse_list = re.findall(r'\[([^]]*)\]',textfile)
    print(verse_list)
    with open('jihad_verses.txt', 'r', encoding = 'utf-8') as verses:
        verses = verses.read()
        verses = verses.splitlines()
        for reference in verses:
            reference = reference.split()
            #print(reference)
            for item in verse_list:
                #print(item)
                if len(reference) > 1:
                    if reference[0] == item or reference[1] == item:
                        print(reference[1])
                   

        
            
            
        
            
                
                
            
            
            
            
        
