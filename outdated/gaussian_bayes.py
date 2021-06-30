from math import sqrt
from math import exp
from math import pi
import numpy as np
import pandas as pd
import string
import unicodedata
from scipy.sparse import csr_matrix
from collections import deque
from pdb import  set_trace as st


def accuracy(y, y_hat):
    eqs = 0
    for a, b in zip(y, y_hat):
        if a == b:
            eqs += 1

    return eqs / len(y)


def tokenize(doc, ngram_range=(1, 3)):

    collected_ngrams = []
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    doc = filter_non_printable(doc)
    for ng_size in range(ngram_range[0], ngram_range[1] + 1):
        window = deque(maxlen=ng_size)
        for ch in doc:
            window.append(ch)
            collected_ngrams.append(''.join(list(window)))

    return collected_ngrams

    
def categorical_encoder(X, vocabulary=None, binary_vertorizer=False):
    X = list(map(tokenize, X))
    if vocabulary is None:
        vocabulary = list(set(sum([d for d in X], [])))

    word_to_idx = {token: idx + 1 for idx, token in enumerate(vocabulary)}
    docs_idxs = []
    for doc in X:
        d_idxs = []
        for token in doc:
            try:
                d_idxs.append(word_to_idx[token])
            except KeyError:
                pass
        docs_idxs.append(d_idxs)
    row = []
    col = []
    data = []
    for i, r in enumerate(docs_idxs):
        if binary_vertorizer:
            used = []
            for c in r:
                if not c in used:
                    row.append(i)
                    col.append(c)
                    data.append(1.0)
                    used.append(c)
        else:
            for c in r:
                row.append(i)
                col.append(c)
                data.append(1.0)

    X_csr = csr_matrix((data, (row, col)), shape=(len(docs_idxs), len(vocabulary) + 1))

    return X_csr, vocabulary


def filter_non_printable(str):
    printable = {'Lu', 'Ll'}
  
    return ''.join(c for c in str if unicodedata.category(c) in printable or c == ' ')


def prepare_data_RVs(file_url, sample=0.01, languages=['English', 'German', 'French', 'Spanish']):
    """Load dataset from
    https://www.kaggle.com/basilb2s/language-detection
    and convert each n-gram 'w' of characters into a feature, that is
    an item of the sample space of a random variable 'X(w)' representing
    input documents.
    """

    dataset = pd.read_csv(file_url, encoding="utf-8")
    dataset = dataset[dataset.Language.isin(languages)].sample(frac=sample)
    train_data = dataset.iloc[0:int(0.7 * len(dataset.index))]
    Xtrain = train_data.Text
    Ytrain = train_data.Language

    test_data = dataset.iloc[int(0.7 * len(dataset.index)):]
    Xtest = test_data.Text
    Ytest = test_data.Language

    return Xtrain, Ytrain, Xtest, Ytest
 

class GaussianClassifier: 

    def __init__(self, ngram_range=(1, 2)):
        self.ngram_range = ngram_range


    def normalize(self, X, E=None, S=None): 
        m, n = X.shape
        if E is None or S is None:
            self.expectation = X.mean(axis=0)
            self.standard_dv = X.std(axis=0)

        X = (X - self.expectation)/self.standard_dv
        
        return X


    def separate_by_class(self, dataset):
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if (class_value not in separated):
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated
    

    def summarize_dataset(self, dataset):
        means = dataset.mean(0)
        stdvs = dataset.std(0)
        lens = [len(dataset)] * dataset.shape[1]
        summaries = list(zip(means, stdvs, lens))
        del(summaries[-1])
        zero_var = [i for i, v in enumerate(summaries) if v[1] <= 0]

        return summaries, zero_var
    

    def summarize_by_class(self, dataset):
        separated = self.separate_by_class(dataset)
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.summarize_dataset(np.vstack(rows))

        return summaries


    def fit(self, X, Y):
        X = self.normalize(X)
        dataset = np.hstack([X, Y.reshape(-1, 1)])
        self.parameters = self.summarize_by_class(dataset)
    
        return self

    def calculate_probability(self, x, mean, stdev):
        if stdev <= 0.0:
            stdev = 0.0000001
        exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
        
        return (1 / (sqrt(2 * pi) * stdev)) * exponent
    

    def calculate_class_probabilities(self, summaries, row, zerov):
        total_rows = sum([summaries[label][0][0][2] for label in summaries])
        probabilities = dict()
        for class_value, sums in summaries.items():
            class_summaries = sums[0]
            probabilities[class_value] = summaries[class_value][0][0][2]/float(total_rows)
            for i in range(len(class_summaries)):
                if not i in zerov:
                    mean, stdev, _ = class_summaries[i]
                    probabilities[class_value] *= self.calculate_probability(row[i], mean, stdev)
        return probabilities
    

    def posterior(self, summaries, row, zerov):
        probabilities = self.calculate_class_probabilities(summaries, row, zerov)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label


    def predict(self, X):
        zerovs = [set(z[1]) for s, z in self.parameters.items()]
        zerovs = set.intersection(*zerovs)
        labels = []
        X = self.normalize(X, E=self.expectation, S=self.standard_dv)
        for row in X:
            labels.append(self.posterior(self.parameters, row, zerov=zerovs))

        return labels


# MAIN
ngrams = (1, 2)
training_portion = 1.0
languages = ['English', 'French']
url = "https://raw.githubusercontent.com/iarroyof/ukp_app/main/Language%20Detection.csv"

# Load train and test data for three languages:
X_train, Y_train, X_test, Y_test = prepare_data_RVs(url, sample=training_portion, languages=languages)
X_train, vocabulary = categorical_encoder(X_train)
X_train = X_train.toarray()
X_test, _ = categorical_encoder(X_test, vocabulary=vocabulary)
X_test = X_test.toarray()

label2class = {l: c for l, c in enumerate(set(Y_train))}
class2label = dict(map(reversed, label2class.items()))

Yl_train = np.array([class2label[c] for c in Y_train])
Yl_test = np.array([class2label[c] for c in Y_test])

gaussian = GaussianClassifier(ngram_range=ngrams)

gaussian.fit(X_train, Yl_train)
y_hat = gaussian.predict(X_test)

result = pd.DataFrame({"Y_hat": [label2class[label] for label in y_hat],
                        "Y": [label2class[label] for label in Yl_test]})

print(result.head(50))
print("\nTest Accuracy: {}".format(accuracy(result.Y_hat, result.Y)))
print("\nClass Balance:")
print(result.Y.value_counts(normalize=True)*100)