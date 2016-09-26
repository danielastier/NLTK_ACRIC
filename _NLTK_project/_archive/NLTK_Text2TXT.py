__author__ = 'Daniela Stier'

# import statements
import pandas as pd

# read preprocessed data, store in lists (clean_data)
reader = open("data_pre_processed.txt", "r")
labels = list()
clean_data = list()
for line in reader.readlines():
    line_split = line.split("\t")
    if line_split[0] == "nor":
        labels.append(0)
    elif line_split[0] == "rad":
        labels.append(1)
    sentence = ""
    for sent in line_split[1:-1]:
        sent_split = sent.split(", ")
        for word in sent_split:
            if word.startswith("['"):
                if not word[2:word.index("/")] == '"':
                    sentence += word[2:word.index("/")] + " "
            elif word.endswith("']"):
                if not word[2:word.index("/")] == '"':
                    sentence += word[2:word.index("/")]
            else:
                if not word[2:word.index("/")] == '"':
                    sentence += word[1:word.index("/")] + " "
    clean_data.append(sentence)

# store vectorized data in 'data_vectors.csv' file, including labels at first position
data_matrix = pd.concat([pd.DataFrame(labels), pd.DataFrame(clean_data)], axis=1)
data_matrix.to_csv('_data_vectors/raw_text.txt', index=False, delimiter='+#')