from collections import deque
import numpy as np
import pandas as pd
import string
import unicodedata
from scipy.sparse import csr_matrix, vstack

from pdb import set_trace as st


def accuracy(y, y_hat):
    eqs = 0
    for a, b in zip(y, y_hat):
        if a == b:
            eqs += 1

    return eqs / len(y)


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


def fill_RVs(X_data, Y_data=None):
        X_data = X_data.apply(tokenize)
        X_data = X_data.apply(set)
        X_data = X_data.apply(list)

        X = []
        Y = []
        if Y_data is None:
            for x in X_data:
                X += x

            return X
        else:
            for x, y in zip(X_data, Y_data):
                Y += [y] * len(x)
                X += x

            return X, Y

class GaussianClassifier:

    def __init__(self, ngram_range=(1, 4), binary_vertorizer=True):
        self.ngram_range = ngram_range
        self.binary_vertorizer = binary_vertorizer


    def var(self, x, axis=None, mean=None):
        x_2 = x.copy()
        x_2.data **= 2

        return x_2.mean(axis) - np.square(x.mean(axis) if mean is None else mean)


    def std(self, x, axis=None, mean=None):

        return np.squeeze(np.asarray(np.sqrt(self.var(x, axis, mean))))


    def gaussian(self, x, mean, std):
        arg = -((x - mean) ** 2 / (2 * std ** 2 ))
        Z = np.sqrt(2 * np.pi) * std

        return (1 / Z) * np.exp(arg)


    def tokenize(self, doc):

        collected_ngrams = []
        doc = doc.translate(str.maketrans('', '', string.punctuation))
        doc = filter_non_printable(doc)
        for ng_size in range(self.ngram_range[0], self.ngram_range[1] + 1):
            window = deque(maxlen=ng_size)
            for ch in doc:
                window.append(ch)
                collected_ngrams.append(''.join(list(window)))

        return collected_ngrams


    def categorical_encoder(self, X, vocabulary=None):
        X = list(map(self.tokenize, X))
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
            if self.binary_vertorizer:
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


    def fit(self, X, Y):

        Xs, self.vocabulary = self.categorical_encoder(X)
        self.languages = np.unique(Y)
        samples_by_lang = {l:[] for l in self.languages}

        for x, l in zip(Xs, Y):
            samples_by_lang[l].append(x)

        self.means = {}
        self.stds = {}
        self.margs = {}
        for l, lmxs in samples_by_lang.items():
            mtx = vstack(lmxs)
            self.margs[l] = len(lmxs) / Xs.shape[0]
            self.means[l] = np.squeeze(np.asarray(mtx.mean(axis=0)))
            self.stds[l] = self.std(mtx, axis=0, mean=self.means[l])

        return self


    def posterior(self, x):
        self.PYX = {l: self.margs[l] for l in self.languages}
        for l in languages:
            for d, i in enumerate(x.indices):
                p_x = self.gaussian(x.data[d], self.means[l][i], self.stds[l][i])
                self.PYX[l] *= p_x

        return max(self.PYX, key=self.PYX.get)


    def predict(self, X):
        X, _ = self.categorical_encoder(X, vocabulary=self.vocabulary)
        Y_hat = [self.posterior(x) for x in X]

        return Y_hat


# MAIN
ngrams = (1, 3)
training_portion = 0.1
languages = ['English', 'Spanish']
url = "https://raw.githubusercontent.com/iarroyof/ukp_app/main/Language%20Detection.csv"

# Load train and test data for three languages:
X_train, Y_train, X_test, Y_test = prepare_data_RVs(url, sample=training_portion, languages=languages)
gaussian = GaussianClassifier(ngram_range=ngrams, binary_vertorizer=True)
gaussian.fit(X_train, Y_train)

Y_hat = gaussian.predict(X_test)

print(Y_hat)

st()