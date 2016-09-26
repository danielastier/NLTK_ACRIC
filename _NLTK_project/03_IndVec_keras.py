__author__ = 'Daniela Stier'

# import statements
import numpy as np
import pandas as pd
import math as mt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Convolution1D, MaxPooling1D, Flatten
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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
                sentence += word[2:word.index("/")] + " "
            elif word.endswith("']"):
                sentence += word[2:word.index("/")]
            else:
                sentence += word[1:word.index("/")] + " "
    clean_data.append(sentence)

max_doc = max([doc for doc in clean_data])

# set variables
nb_words = 50
max_doc_len = 500

# keras tokenizer: only consider 5000 most frequent words in all documents
# output: list of word indexes for each document
tokenizer = Tokenizer(nb_words=nb_words)
tokenizer.fit_on_texts(clean_data)
sequences = tokenizer.texts_to_sequences(clean_data)
word_index = tokenizer.word_index

# keras padding: truncate sequences to the number of words of the longest document
clean_data = pad_sequences(sequences, maxlen=max_doc_len)

# store vectorized data in 'data_vectors.csv' file, including labels at first position
data_matrix = pd.concat([pd.DataFrame(labels), pd.DataFrame(clean_data)], axis=1)
data_matrix.to_csv('_data_vectors/index_vector_02.csv', index=False, delimiter=',')

# input data - only for approach employing own word vector
clean_data = pd.read_csv('_data_vectors/index_vector_02.csv', sep=',')
clean_data = clean_data.iloc[np.random.permutation(len(clean_data))]
data_features = clean_data.iloc[:, 1:].as_matrix()
labels = clean_data.iloc[:, 0].as_matrix()


print("################### INDEX VECTOR: Logistic Regression (10-fold cross validation) ###################")
print("labels", len(labels))
print("data_features", data_features.shape)

# create and fit the model
lrm_cv_index = LogisticRegression(penalty='l2', C=1000000)#C=1/50)
lrm_cv_index.fit(data_features, labels)

# report resulting accuracies
cv_scores_lrm_index = cross_val_score(lrm_cv_index, data_features, labels, cv=10)
print("cross-validation scores: ", cv_scores_lrm_index)
cv_mean_lrm_index = np.mean(cv_scores_lrm_index)
print("mean accuracy cv: ", cv_mean_lrm_index)
sterr_lrm_index = np.std(cv_scores_lrm_index)/(mt.sqrt(len(cv_scores_lrm_index)))
print("standard error: ", sterr_lrm_index)


print("################### EXERCISE 02: MLP (10-fold cross validation) ###################")
print("labels_02", len(labels))
print("data_features_02", data_features.shape)

# create and fit the model
hidden_dims = 100 # tested for [25, 50, 75, 100]
kfold_mlp = StratifiedKFold(y=labels, n_folds=10, shuffle=True, random_state=True)
cv_scores_mlp = []
for i, (train, test) in enumerate(kfold_mlp):
    mlp_cv = Sequential()
    mlp_cv.add(Dense(input_dim=data_features.shape[1], output_dim=hidden_dims, activation='relu', init='uniform'))
    mlp_cv.add(Dropout(0.5))
    mlp_cv.add(Dense(output_dim=1, activation='sigmoid', init='uniform'))
    # compile model
    mlp_cv.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # fit model
    mlp_cv.fit(data_features[train], labels[train], nb_epoch=2, verbose=0)
    # evaluate model
    cv_scores = mlp_cv.evaluate(data_features[test], labels[test], verbose=0)
    cv_scores_mlp.append(cv_scores[1] * 100)

# report resulting accuracies
print("cross-validation scores: ", cv_scores_mlp)
cv_mean_mlp = np.mean(cv_scores_mlp)
print("mean accuracy cv: ", cv_mean_mlp)
sterr_mlp = np.std(cv_scores_mlp)/(mt.sqrt(len(cv_scores_mlp)))
print("standard error: ", sterr_mlp)


print("################### EXERCISE 03: CNN (10-fold cross validation) ###################")
print("labels_03", len(labels))
print("data_features_03", data_features.shape)

# set variables
max_features = len(word_index)+1 # vocabulary: number of features/unique tokens after limitation of data to most frequent words
max_len = max_doc_len # maximum document/sequence length - all documents are padded to this length
embedding_dims = 100 # vocabulary mapped onto x dimensions
feature_maps = 25 # number of feature maps for each filter size
filter_size = 5 # size of applied filter, covering at least bigrams = 2
hidden_dims = 50
batch_size = 16

### create and fit the model
kfold_cnn = StratifiedKFold(y=labels, n_folds=10, shuffle=True, random_state=True)
cv_scores_cnn = []
for i, (train, test) in enumerate(kfold_cnn):
    cnn_cv = Sequential()
    cnn_cv.add(Embedding(max_features, embedding_dims, input_length=max_len, dropout=0.5))
    cnn_cv.add(Convolution1D(nb_filter=feature_maps, filter_length=filter_size, activation='relu'))
    cnn_cv.add(MaxPooling1D(pool_length=cnn_cv.output_shape[2]))
    cnn_cv.add(Flatten())
    cnn_cv.add(Dense(hidden_dims, activation='relu'))
    cnn_cv.add(Dropout(0.2))
    cnn_cv.add(Dense(1, activation='sigmoid'))
    # compile model
    cnn_cv.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    # fit model
    cnn_cv.fit(data_features[train], labels[train], batch_size=batch_size, nb_epoch=2, verbose=0)#, validation_data=(data_features_03[test], labels_03[test]))
    # evaluate model
    cv_scores = cnn_cv.evaluate(data_features[test], labels[test], verbose=0)
    cv_scores_cnn.append(cv_scores[1] * 100)

# report resulting accuracies
print("cross-validation scores: ", cv_scores_cnn)
cv_mean_cnn = np.mean(cv_scores_cnn)
print("mean accuracy cv: ", cv_mean_cnn)
sterr_cnn = np.std(cv_scores_cnn)/(mt.sqrt(len(cv_scores_cnn)))
print("standard error: ", sterr_cnn)