__author__ = 'Daniela Stier'

# IMPORT STATEMENTS
import numpy as np
import pandas as pd

clean_data = pd.read_csv('_data_vectors/main_data_vectors.csv', sep=',', header=0)
header = clean_data.columns

# store data in ARFF file format
#with open("nltk_data.arff", 'a', newline='') as w:
with open("nltk_features.arff", 'a', newline='') as w:
    # store data in 'cl4lt_data.arff' file
    w.write("% 1. Title: NLTK Database\n")
    w.write("% Group Project: Classifying Radical Ideologies\n")
    w.write("%\n")
    w.write("% 2. Sources:\n")
    w.write("%      (a) Members: L. Hiller, D. Stier\n")
    w.write("%      (b) University of Tübingen\n")
    w.write("%      (c) Date: August, 2016\n")
    w.write("%\n")
    w.write("\n")
    w.write("@relation rad-nor\n")
    w.write("@attribute" + " " + header[0] + " " + "{nor,rad}\n")
    for head in range(1, len(header)):
        w.write("@attribute" + " " + header[head] + " " + "numeric\n")
    w.write("\n")
    w.write("@data\n")
w.close()
#clean_data.to_csv("nltk_data.arff", mode='a', header=False, index=False, sep = ",")
clean_data.to_csv("nltk_features.arff", mode='a', header=False, index=False, sep = ",")
