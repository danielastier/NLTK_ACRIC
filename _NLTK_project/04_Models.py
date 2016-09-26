__author__ = 'Daniela Stier'

# IMPORT STATEMENTS
import csv
import itertools
import numpy as np
import pandas as pd
import nltk
import math as mt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Convolution1D, MaxPooling1D, Flatten

############################################# APPLICATIONAL PART #############################################

# merging data vectors, stored in 'main_data_vectors.csv'
#csv_names = ["_data_vectors_wider/sent_vector.csv", "_data_vectors_wider/word_vector.csv", "_data_vectors_wider/char_vector.csv", "_data_vectors_wider/puncts_vector.csv", "_data_vectors_wider/sent01_vector.csv", "_data_vectors_wider/sent02_vector.csv"]
csv_names = ["_data_vectors/sent_vector.csv", "_data_vectors/word_vector.csv", "_data_vectors/char_vector.csv", "_data_vectors/puncts_vector.csv", "_data_vectors/sent01_vector.csv", "_data_vectors/sent02_vector.csv"]
readers = [csv.reader(open(r, 'r')) for r in csv_names]
#writer = csv.writer(open('_data_vectors_wider/main_data_vectors.csv', 'w'))
writer = csv.writer(open('_data_vectors/main_data_vectors.csv', 'w'))
for row in zip(*readers):
    writer.writerow(list(itertools.chain.from_iterable(row)))

# input data - FEATURE VECTOR
#clean_data = pd.read_csv('_data_vectors_wider/main_data_vectors.csv', sep=',', header=1)
clean_data = pd.read_csv('_data_vectors/main_data_vectors.csv', sep=',', header=1)
clean_data = clean_data.iloc[np.random.permutation(len(clean_data))]
clean_data.replace(['nor', 'rad'], [1, 0], inplace=True)
data_features = clean_data.iloc[:, 1:].as_matrix()
labels = clean_data.iloc[:, 0].as_matrix()
labels = np.array(labels)
data_features = np.array(data_features)

# input data - INDEX VECTOR
#clean_data_index = pd.read_csv('_data_vectors_wider/index_vector.csv', sep=',', header=1)
clean_data_index = pd.read_csv('_data_vectors/index_vector.csv', sep=',', header=1)
clean_data_index = clean_data_index.iloc[np.random.permutation(len(clean_data_index))]
clean_data_index.replace(['nor', 'rad'], [1, 0], inplace=True)
data_features_index = clean_data_index.iloc[:, 1:].as_matrix()
labels_index = clean_data_index.iloc[:, 0].as_matrix()
labels_index = np.array(labels_index)
data_features_index = np.array(data_features_index)


print("################### Logistic Regression C=1000000 (10-fold cross validation) ###################")
print("labels", len(labels))
print("data_features", data_features.shape)

# create and fit the model
lrm_cv = LogisticRegression(penalty='l2', C=1000000)#C=1/50)
lrm_cv.fit(data_features, labels)

# report resulting accuracies
cv_scores_lrm = cross_val_score(lrm_cv, data_features, labels, cv=10)
print("cross-validation scores: ", cv_scores_lrm)
cv_mean_lrm = np.mean(cv_scores_lrm)
print("mean accuracy cv: ", cv_mean_lrm)
sterr_lrm = np.std(cv_scores_lrm)/(mt.sqrt(len(cv_scores_lrm)))
print("standard error: ", sterr_lrm)


# print("################### INDEX VECTOR: Logistic Regression (10-fold cross validation) ###################")
# print("labels", len(labels_index))
# print("data_features", data_features_index.shape)
#
# # create and fit the model
# lrm_cv_index = LogisticRegression(penalty='l2', C=1000000)#C=1/50)
# lrm_cv_index.fit(data_features_index, labels_index)
#
# # report resulting accuracies
# cv_scores_lrm_index = cross_val_score(lrm_cv_index, data_features_index, labels_index, cv=10)
# print("cross-validation scores: ", cv_scores_lrm_index)
# cv_mean_lrm_index = np.mean(cv_scores_lrm_index)
# print("mean accuracy cv: ", cv_mean_lrm_index)
# sterr_lrm_index = np.std(cv_scores_lrm_index)/(mt.sqrt(len(cv_scores_lrm_index)))
# print("standard error: ", sterr_lrm_index)
#
#
# print("################### MLP (10-fold cross validation) ###################")
# print("labels", len(labels))
# print("data_features", data_features.shape)
#
# # create and fit the model
# hidden_dims = 100 # tested for [25, 50, 75, 100]
# kfold_mlp = StratifiedKFold(y=labels, n_folds=10, shuffle=True, random_state=True)
# cv_scores_mlp = []
# for i, (train, test) in enumerate(kfold_mlp):
#     mlp_cv = Sequential()
#     mlp_cv.add(Dense(input_dim=data_features.shape[1], output_dim=hidden_dims, activation='relu', init='uniform'))
#     mlp_cv.add(Dropout(0.5))
#     mlp_cv.add(Dense(output_dim=1, activation='sigmoid', init='uniform'))
#     # compile model
#     mlp_cv.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#     # fit model
#     mlp_cv.fit(data_features[train], labels[train], nb_epoch=2, verbose=0)
#     # evaluate model
#     cv_scores = mlp_cv.evaluate(data_features[test], labels[test], verbose=0)
#     cv_scores_mlp.append(cv_scores[1] * 100)
#
# # report resulting accuracies
# print("cross-validation scores: ", cv_scores_mlp)
# cv_mean_mlp = np.mean(cv_scores_mlp)
# print("mean accuracy cv: ", cv_mean_mlp)
# sterr_mlp = np.std(cv_scores_mlp)/(mt.sqrt(len(cv_scores_mlp)))
# print("standard error: ", sterr_mlp)
#
#
# print("################### MLP (10-fold cross validation) ###################")
# print("labels", len(labels_index))
# print("data_features", data_features_index.shape)
#
# # create and fit the model
# hidden_dims_index = 100 # tested for [25, 50, 75, 100]
# kfold_mlp_index = StratifiedKFold(y=labels_index, n_folds=10, shuffle=True, random_state=True)
# cv_scores_mlp_index = []
# for i, (train_index, test_index) in enumerate(kfold_mlp_index):
#     mlp_cv_index = Sequential()
#     mlp_cv_index.add(Dense(input_dim=data_features_index.shape[1], output_dim=hidden_dims_index, activation='relu', init='uniform'))
#     mlp_cv_index.add(Dropout(0.5))
#     mlp_cv_index.add(Dense(output_dim=1, activation='sigmoid', init='uniform'))
#     # compile model
#     mlp_cv_index.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#     # fit model
#     mlp_cv_index.fit(data_features_index[train_index], labels_index[train_index], nb_epoch=2, verbose=0)
#     # evaluate model
#     cv_scores_index = mlp_cv_index.evaluate(data_features_index[test_index], labels_index[test_index], verbose=0)
#     cv_scores_mlp_index.append(cv_scores_index[1] * 100)
#
# # report resulting accuracies
# print("cross-validation scores: ", cv_scores_mlp_index)
# cv_mean_mlp_index = np.mean(cv_scores_mlp_index)
# print("mean accuracy cv: ", cv_mean_mlp_index)
# sterr_mlp_index = np.std(cv_scores_mlp_index)/(mt.sqrt(len(cv_scores_mlp_index)))
# print("standard error: ", sterr_mlp_index)
#
#
# print("################### CNN (10-fold cross validation) ###################")
# print("labels", len(labels))
# print("data_features", data_features.shape)
#
# # set variables
# # FOR INDEX VECTOR: calculate max value for dataframe, insert in max_features: int(dataframe.values.max())+1
# max_features = int(clean_data.values.max())+1 # vocabulary size
# max_len = data_features.shape[1] # maximum document/sequence length
# embedding_dims = 100 # vocabulary mapped onto x dimensions
# feature_maps = 25 # number of feature maps for each filter size
# filter_size = 5 # size of applied filter, covering at least bigrams = 2
# hidden_dims = 50
# batch_size = 16
#
# ### create and fit the model
# kfold_cnn = StratifiedKFold(y=labels, n_folds=10, shuffle=True, random_state=True)
# cv_scores_cnn = []
# for i, (train, test) in enumerate(kfold_cnn):
#     cnn_cv = Sequential()
#     cnn_cv.add(Embedding(max_features, embedding_dims, input_length=max_len, dropout=0.5))
#     cnn_cv.add(Convolution1D(nb_filter=feature_maps, filter_length=filter_size, activation='relu'))
#     cnn_cv.add(MaxPooling1D(pool_length=cnn_cv.output_shape[2]))
#     cnn_cv.add(Flatten())
#     cnn_cv.add(Dense(hidden_dims, activation='relu'))
#     cnn_cv.add(Dropout(0.2))
#     cnn_cv.add(Dense(1, activation='sigmoid'))
#     # compile model
#     cnn_cv.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#     # fit model
#     cnn_cv.fit(data_features[train], labels[train], batch_size=batch_size, nb_epoch=2, verbose=0)#, validation_data=(data_features_03[test], labels_03[test]))
#     # evaluate model
#     cv_scores = cnn_cv.evaluate(data_features[test], labels[test], verbose=0)
#     cv_scores_cnn.append(cv_scores[1] * 100)
#
# # report resulting accuracies
# print("cross-validation scores: ", cv_scores_cnn)
# cv_mean_cnn = np.mean(cv_scores_cnn)
# print("mean accuracy cv: ", cv_mean_cnn)
# sterr_cnn = np.std(cv_scores_cnn)/(mt.sqrt(len(cv_scores_cnn)))
# print("standard error: ", sterr_cnn)
#
#
# print("################### INDEX VECTOR: CNN (10-fold cross validation) ###################")
# print("labels", len(labels_index))
# print("data_features", data_features_index.shape)
#
# # set variables
# # FOR INDEX VECTOR: calculate max value for dataframe, insert in max_features: int(dataframe.values.max())+1
# max_features_index = 5001 # vocabulary size
# max_len_index = data_features_index.shape[1] # maximum document/sequence length
# embedding_dims_index = 100 # vocabulary mapped onto x dimensions
# feature_maps_index = 25 # number of feature maps for each filter size
# filter_size_index = 5 # size of applied filter, covering at least bigrams = 2
# hidden_dims_index = 50
# batch_size_index = 16
#
# ### create and fit the model
# kfold_cnn_index = StratifiedKFold(y=labels_index, n_folds=10, shuffle=True, random_state=True)
# cv_scores_cnn_index = []
# for i, (train_index, test_index) in enumerate(kfold_cnn_index):
#     cnn_cv_index = Sequential()
#     cnn_cv_index.add(Embedding(max_features_index, embedding_dims_index, input_length=max_len_index, dropout=0.5))
#     cnn_cv_index.add(Convolution1D(nb_filter=feature_maps_index, filter_length=filter_size_index, activation='relu'))
#     cnn_cv_index.add(MaxPooling1D(pool_length=cnn_cv_index.output_shape[2]))
#     cnn_cv_index.add(Flatten())
#     cnn_cv_index.add(Dense(hidden_dims_index, activation='relu'))
#     cnn_cv_index.add(Dropout(0.2))
#     cnn_cv_index.add(Dense(1, activation='sigmoid'))
#     # compile model
#     cnn_cv_index.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#     # fit model
#     cnn_cv_index.fit(data_features_index[train_index], labels_index[train_index], batch_size=batch_size_index, nb_epoch=2, verbose=0)#, validation_data=(data_features_03[test], labels_03[test]))
#     # evaluate model
#     cv_scores_index = cnn_cv_index.evaluate(data_features_index[test_index], labels_index[test_index], verbose=0)
#     cv_scores_cnn_index.append(cv_scores_index[1] * 100)
#
# # report resulting accuracies
# print("cross-validation scores: ", cv_scores_cnn_index)
# cv_mean_cnn_index = np.mean(cv_scores_cnn_index)
# print("mean accuracy cv: ", cv_mean_cnn_index)
# sterr_cnn_index = np.std(cv_scores_cnn_index)/(mt.sqrt(len(cv_scores_cnn_index)))
# print("standard error: ", sterr_cnn_index)