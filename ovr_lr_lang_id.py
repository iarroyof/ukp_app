# This code corresponds to a Language Identification (LI) approach which uses a 
# multiclass Logistic Regression classifier. According to [1], this classifier
# has been used in a variety of LI datasets in the literature. I selected and
# reimplemented this approach because it has reported high performance in the 
# LI task despite of its simplicity.
#
#                   -- Ignacio Arroyo-Fernandez --
# -----------------------------------------------------------------------------
# [1] Jauhiainen, Tommi, et al. "Automatic language identification in texts: A
#       survey." Journal of Artificial Intelligence Research 65 (2019): 675-782
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from collections import deque
from scipy.sparse import csr_matrix
import string
import unicodedata


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

    X_csr = csr_matrix((data, (row, col)),
                        shape=(len(docs_idxs), len(vocabulary) + 1))

    return X_csr, vocabulary


def filter_non_printable(str):
    printable = {'Lu', 'Ll'}
  
    return ''.join(
        c for c in str if unicodedata.category(c) in printable or c == ' ')


def prepare_data(file_url, sample=0.01,
                        languages=['English', 'German', 'French', 'Spanish']):
    """
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


def accuracy(y, y_hat):
    eqs = 0
    for a, b in zip(y, y_hat):
        if a == b:
            eqs += 1

    return eqs / len(y)


class OVRClassifier:

    def __init__(self, BinaryClassifier):
        self.BinaryClassifier = BinaryClassifier

    def make_datsets(self, X, Y):
        
        self.classes = set(Y)
        datasets = {c: [] for c in self.classes}
        for x, y in zip(X, Y):
            datasets[y].append(x)

        return datasets


    def make_ovr_dataset(self, class_, datasets):
        X_positive = np.vstack(datasets[class_])
        Y_positive = np.array([1] * X_positive.shape[0])
        X_negative = [np.vstack(datasets[l])
                                    for l in datasets.keys() if class_ != l]
        X_negative = np.vstack(X_negative)
        Y_negative = np.array([0] * X_negative.shape[0])
        X = np.vstack([X_positive, X_negative])
        Y = np.vstack([Y_positive.reshape(-1, 1), Y_negative.reshape(-1, 1)])

        return X, Y


    def fit(self, X, Y):
        datasets = self.make_datsets(X, Y)
        self.classifiers = {}

        for c in self.classes:
            X, Y =self.make_ovr_dataset(class_=c, datasets=datasets)
            clf = self.BinaryClassifier()
            self.classifiers[c] = clf.fit(X, Y)

        return self


    def predict(self, X):
        predictions = np.zeros((len(self.classes), X.shape[0]))
        for c in self.classes:
            predictions[c, :] = self.classifiers[c].predict(X).reshape(1, -1)

        return predictions.argmax(0)


class LogisticRegressionClassifier:

    def __init__(self, normalize=False, batch_size=10, n_epochs=100,
                                    learning_rate=0.01, predict_proba=True):
        self.normalize = normalize
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.predict_proba = predict_proba


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))


    def loss_function(self, y, y_hat):
        cross_entropy = -np.mean(
                            y * (np.log(y_hat)) - (1 - y) * np.log(1 - y_hat))

        return cross_entropy


    def evaluate_gradients(self, X, y, y_hat):
        N = X.shape[0]

        Dw = (1 / N) * np.dot(X.T, (y_hat - y))
        Db = (1 / N) * np.sum((y_hat - y)) 

        return Dw, Db


    def normalize_data(self, X):
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        return X


    def fit(self, X, y):

        m, n = X.shape
        self.W = np.zeros((n, 1))
        self.b = 0
        y = y.reshape(m, 1)

        if self.normalize:
            X = self.normalize_data(X)

        for epoch in range(self.n_epochs):
            for i in range((m - 1)//self.batch_size + 1):
                start = i * self.batch_size
                end = start + self.batch_size
                x_batch = X[start:end]
                y_batch = y[start:end]
                
                y_hat = self.sigmoid(np.dot(x_batch, self.W) + self.b)
                D_W, D_b = self.evaluate_gradients(x_batch, y_batch, y_hat)

                self.W -= self.learning_rate * D_W
                self.b -= self.learning_rate * D_b
            
            ce = self.loss_function(
                                y, self.sigmoid(np.dot(X, self.W) + self.b))
            print("Epoch {}/{}  |  Cross Entropy loss: {}".format(
                                                    epoch, self.n_epochs, ce))
        
        return self


    def predict(self, X):
        if self.normalize:
            X = self.normalize_data(X)

        preds = self.sigmoid(np.dot(X, self.W) + self.b)
        if self.predict_proba:
            return preds

        pred_class = [1 if i > 0.5 else 0 for i in preds]

        return np.array(pred_class)



# MAIN
ngrams = (1, 2)
training_portion = 1.0
languages = ['English', 'French', 'German']
# Load dataset from https://www.kaggle.com/basilb2s/language-detection
# I uploaded it to my git repo:
url = ("https://raw.githubusercontent.com/"
        "iarroyof/ukp_app/main/Language%20Detection.csv")

# Prepare train and test data for three languages. Each dcoument is represented
# first as a list of character n-gram tokens. Then the documents are encoded
# in binary vectors. You can test the code for any subset of the available 
# languages in the dataset
X_train, Y_train, X_test, Y_test = prepare_data(
                            url, sample=training_portion, languages=languages)
X_train, vocabulary = categorical_encoder(X_train)
X_train = X_train.toarray()
X_test, _ = categorical_encoder(X_test, vocabulary=vocabulary)
X_test = X_test.toarray()

# Convert classes to integers and create maps to show results
label2class = {l: c for l, c in enumerate(set(Y_train))}
class2label = dict(map(reversed, label2class.items()))

Yl_train = np.array([class2label[c] for c in Y_train])
Yl_test = np.array([class2label[c] for c in Y_test])

# Create and fit One-Versus-Rest Logistic Regression classifier using default
# hyperparameters
ovr_clf = OVRClassifier(LogisticRegressionClassifier)
ovr_clf.fit(X_train, Yl_train)

# Make predictions on test data
y_hat = ovr_clf.predict(X_test)

print("\nSample of predictions on the test set:\n")
result = pd.DataFrame({"Y": [label2class[label] for label in Yl_test],
                        "Y_hat": [label2class[label] for label in y_hat]})
print(result.head(50))

# Print test performace results
print("\nTest Accuracy: {}".format(accuracy(Yl_test, y_hat)))