import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
import re
import string
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import f1_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import csv



# load data

test_samples_data = np.genfromtxt('test_samples.txt', encoding='utf-8', dtype=None, delimiter='\t',
                                  names=('col1', 'col2'))
test_samples1 = test_samples_data['col2']
test_samples_0 = test_samples_data['col1']
# test_samples = np.array(test_samples)
# test_samples_0 = np.array(test_samples_0)

train_labels_data = np.genfromtxt('train_labels.txt', encoding='utf-8', dtype=None, delimiter='\t',
                                  names=('col1', 'col2'))
train_labels = train_labels_data['col2']
# train_labels = np.array(train_labels)

train_samples_data = np.genfromtxt('train_samples.txt', encoding='utf-8', dtype=None, delimiter='\t',
                                   names=('col1', 'col2'))
train_samples1 = train_samples_data['col2']
# train_samples = np.array(train_samples)


validation_samples_data = np.genfromtxt('validation_samples.txt', encoding='utf-8', dtype=None, delimiter='\t',
                                  names=('col1', 'col2'))
validation_samples1 = validation_samples_data['col2']
validation_samples_0 = validation_samples_data['col1']

validation_labels_data = np.genfromtxt('validation_labels.txt', encoding='utf-8', dtype=None, delimiter='\t',
                                  names=('col1', 'col2'))
validation_labels = validation_labels_data['col2']

# clean data
# translator = str.maketrans('', '', string.punctuation)
tokenizer = RegexpTokenizer(r'\w+')

train_samples = []

for i in range(len(train_samples1)):
    train_samples1[i] = train_samples1[i].replace('$NE$', '')
    #train_samples1[i] = train_samples1[i].replace('„', '')
    #train_samples1[i] = train_samples1[i].replace('”', '')
    # train_samples[i] = train_samples[i].translate(translator)

    result = tokenizer.tokenize(train_samples1[i])
    train_samples.append(result)

test_samples = []

for i in range(len(test_samples1)):
    test_samples1[i] = test_samples1[i].replace('$NE$', '')
    #test_samples1[i] = test_samples1[i].replace('„', '')
    #test_samples1[i] = test_samples1[i].replace('”', '')
    # train_samples[i] = train_samples[i].translate(translator)

    result = tokenizer.tokenize(test_samples1[i])
    test_samples.append(result)


validation_samples = []

for i in range(len(validation_samples1)):
    validation_samples1[i] = validation_samples1[i].replace('$NE$', '')
    #test_samples1[i] = test_samples1[i].replace('„', '')
    #test_samples1[i] = test_samples1[i].replace('”', '')
    # train_samples[i] = train_samples[i].translate(translator)

    result = tokenizer.tokenize(validation_samples1[i])
    validation_samples.append(result)

# print(validation_samples)
# print(test_samples)


# print(train_labels)

# print(train_samples[:, 1])


class Bag_of_words:

    def __init__(self):
        self.words = []
        self.vocabulary_length = 0

    def build_vocabulary(self, data):
        for document in data:
            for word in document:
                word = word.lower()
                if word not in self.words:
                    self.words.append(word)

        self.vocabulary_length = len(self.words)
        self.words = np.array(self.words)

    def get_features(self, data):
        features = np.zeros((len(data), self.vocabulary_length))

        for document_idx, document in enumerate(data):
            for word in document:
                if word in self.words:
                    features[document_idx, np.where(self.words == word)[0][0]] += 1
        return features


nr = 0

if len(test_samples) > len(train_samples):
    nr = len(train_samples)
else:
    nr = len(test_samples)
nr = 150


def normalize_data(train_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')

    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')

    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        return (scaled_train_data, scaled_test_data)
    else:
        print("No scaling was performed. Raw data is returned.")
        return (train_data, test_data)




with open('file.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'label'])

    for i in range(43):

        bow_model = Bag_of_words()
        bow_model.build_vocabulary(train_samples[61 * i: 61 * (i + 1)])
        train_features = bow_model.get_features(train_samples[61 * i: 61 * (i + 1)])
        test_features = bow_model.get_features(test_samples[61 * i: 61 * (i + 1)])
        # print(train_features.shape)
        # print(test_features.shape)
        scaled_train_data, scaled_test_data = normalize_data(train_features, test_features, type='l2')

        svm_model = svm.SVC(C=100, kernel='linear')
        svm_model.fit(scaled_train_data, train_labels[61 * i: 61 * (i + 1)])
        predicted_labels_svm = svm_model.predict(scaled_test_data)

        for j in range(len(predicted_labels_svm)):
            writer.writerow([test_samples_0[61 * i + j], predicted_labels_svm[j]])


#print(validation_samples)

pred = []

for i in range(32):

    bow_model = Bag_of_words()
    bow_model.build_vocabulary(train_samples[83 * i: 83 * (i + 1)])
    train_features = bow_model.get_features(train_samples[83 * i: 83 * (i + 1)])
    test_features = bow_model.get_features(validation_samples[83 * i: 83 * (i + 1)])
    # print(train_features.shape)
    # print(test_features.shape)
    scaled_train_data, scaled_test_data = normalize_data(train_features, test_features, type='l2')

    svm_model = svm.SVC(C=100, kernel='linear')
    svm_model.fit(scaled_train_data, train_labels[83 * i: 83 * (i + 1)])
    predicted_labels_svm = svm_model.predict(scaled_test_data)
    for j in range(len(predicted_labels_svm)):
        pred.append(predicted_labels_svm[j])

print(f1_score(validation_labels, pred, average="macro"))


def confusion_matrix(y_true, y_pred):
    num_classes = max(len(y_true), len(y_pred) + 1)
    conf_matrix = np.zeros((num_classes, num_classes))

    for i in range(len(y_true)):
        conf_matrix[int(y_true[i]), int(y_pred[i])] += 1
    return conf_matrix


print(confusion_matrix(validation_labels, pred))

