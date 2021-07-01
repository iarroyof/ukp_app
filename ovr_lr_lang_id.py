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
from os import path



def tokenize(doc, ngram_range=(1, 3)):
    """This function tokenizes an input string 'doc' as character ngrams. The
    non-words and nonprintable characters are discarded. 
    """
    collected_ngrams = []
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    doc = filter_non_printable(doc)
    for ng_size in range(ngram_range[0], ngram_range[1] + 1):
        window = deque(maxlen=ng_size)
        for ch in doc:
            window.append(ch)
            collected_ngrams.append(''.join(list(window)))

    return collected_ngrams

    
def categorical_encoder(X, vocabulary=None):
    """This function encodes each document (string) in 'X' as a binary array.
    To encode test data only pass the vocabulary collected from the fisrt 
    call to the function so as to use the same indices for each feature.
    """
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
    for i, roww in enumerate(docs_idxs):   
        used = []
        for c in roww:
            if not c in used:
                row.append(i)
                col.append(c)
                data.append(1.0)
                used.append(c)

    X_csr = csr_matrix((data, (row, col)),
                        shape=(len(docs_idxs), len(vocabulary) + 1))

    return X_csr, vocabulary


def filter_non_printable(str):

    printable = {'Lu', 'Ll'}
  
    return ''.join(
        c for c in str if unicodedata.category(c) in printable or c == ' ')


def prepare_data(file_url, sample=0.01,
                        languages=['English', 'German', 'French', 'Spanish']):
    """This function prepares the input CSV file with two columns ['Text',
    'Language'] and encodes each document as a binary array.
    """
    try:
        dataset = pd.read_csv(file_url, encoding="utf-8")
    except:
        if path.exists("Language_Detection.csv"):
            dataset = pd.read_csv("Language_Detection.csv", encoding="utf-8")
        else:
            print("The download address to the training data is not available,"
                "please try some minutes later. Also you can download the data"
                " from my Github repo:"
                "\nhttps://github.com/iarroyof/ukp_app/blob/main/Language_"
                "Detection.csv"
                "\nPlease tut it into you current directory and execute this"
                " script again.")
    dataset = dataset[dataset.Language.isin(languages)].sample(frac=sample)
    train_data = dataset.iloc[0:int(0.7 * len(dataset.index))]
    Xtrain = train_data.Text
    Ytrain = train_data.Language

    test_data = dataset.iloc[int(0.7 * len(dataset.index)):]
    Xtest = test_data.Text
    Ytest = test_data.Language

    Xtrain, vocabulary = categorical_encoder(Xtrain)
    Xtrain = Xtrain.toarray()
    Xtest, _ = categorical_encoder(Xtest, vocabulary=vocabulary)
    Xtest = Xtest.toarray()

    return Xtrain, Ytrain, Xtest, Ytest, vocabulary


def accuracy(Y, Y_hat):

    eqs = 0
    for a, b in zip(Y, Y_hat):
        if a == b:
            eqs += 1

    return eqs / len(Y)


class OVRClassifier:
    """This class implements a One-Versus-Rest classifier using binary
    Logistic Regression models. After trained, the self.classifiers attribite
    is adictionary whose values are the trained models for n_classes.
    To access the parameters of each models it is needed to do
    self.classifiers[label].beta
    self.classifiers[label].beta_0
    """
    def __init__(self, BinaryClassifier):

        self.BinaryClassifier = BinaryClassifier


    def make_datsets(self, X, Y):
        """This function split the dataset into n_classes subsets.
        """
        self.classes = set(Y)
        datasets = {c: [] for c in self.classes}
        for x, y in zip(X, Y):
            datasets[y].append(x)

        return datasets


    def make_ovr_dataset(self, class_, datasets):
        """This function splits the dataset into n_classes subsets and combines
        them to form n_classes OVR learning problems. 
        """
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
        """This function splits the dataset into n_classes subsets and combines
        them to train separately n_classes Logistic Regression models in a OVR
        fashion.
        """
        datasets = self.make_datsets(X, Y)
        self.classifiers = {}

        for c in self.classes:
            print("\nTraining classifier for class {}\n".format(c))
            X, Y =self.make_ovr_dataset(class_=c, datasets=datasets)
            clf = self.BinaryClassifier()
            self.classifiers[c] = clf.fit(X, Y)

        return self


    def predict(self, X):
        """This function takes the trained models for each OVR learning problem
        and predicts their corresponding binomial probability distributions. 
        """
        predictions = np.zeros((len(self.classes), X.shape[0]))
        for c in self.classes:
            predictions[c, :] = self.classifiers[c].predict(X).reshape(1, -1)

        return predictions.argmax(0)


class LogisticRegressionClassifier:
    """This class creates a Logistic Regression model and trains it using
    gradient descent algorithm.
    """
    def __init__(self, batch_size=32, n_epochs=100,
                                    learning_rate=0.01, predict_proba=True):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.predict_proba = predict_proba


    def sigmoid(self, x):

        return 1.0 / (1.0 + np.exp(-x))


    def loss_function(self, Y, Y_hat):
        """This function calculates the cross entropy loss fro training and
        predicted labels.
        """
        cross_entropy = -np.mean(
                            Y * (np.log(Y_hat)) - (1 - Y) * np.log(1 - Y_hat))

        return cross_entropy


    def evaluate_gradients(self, X, Y, Y_hat):
        """This function evaluates precomputed gradients of the loss function.
        """
        N = X.shape[0]
        G_beta = (1 / N) * np.dot(X.T, (Y_hat - Y))
        G_beta_0 = (1 / N) * np.sum((Y_hat - Y)) 

        return G_beta, G_beta_0


    def gradient_descent(self, X, Y, Y_hat):
        """This function evaluates the gradients of the loss function and uses
        them to update the coefficients of the LR model.
        """
        G_beta, G_beta_0 = self.evaluate_gradients(X, Y, Y_hat)
        self.beta -= self.learning_rate * G_beta
        self.beta_0 -= self.learning_rate * G_beta_0


    def logit(self, X):
        """This function computes the logits of the LR model using the current
        coefficients.
        """
        return np.dot(X, self.beta) + self.beta_0


    def batches(self, X):
        """This function splits the input data into equal size batches and
        generates them as items of an interable.
        """
        data_size = X.shape[0]
        for current in range(0, data_size, self.batch_size):
            yield X[current:min(current + self.batch_size, data_size)]


    def fit(self, X, Y):
        """This function fits a Logistic Regression model from data.
        """
        self.beta = np.zeros((X.shape[1], 1))
        self.beta_0 = 0.0
        Y = Y.reshape(-1, 1)

        for epoch in range(self.n_epochs):
            for X_batch, Y_batch in zip(self.batches(X), self.batches(Y)):     
                logits = self.logit(X_batch)
                Y_hat = self.sigmoid(logits)
                self.gradient_descent(X_batch, Y_batch, Y_hat)
            
            ce = self.loss_function(Y_batch, Y_hat)
            print("Epoch {}/{}  |  Cross Entropy loss: {}".format(
                                                    epoch, self.n_epochs, ce))
        
        return self


    def predict(self, X):

        logits = self.logit(X)
        probabilities = self.sigmoid(logits)
        if self.predict_proba:
            return probabilities
        else:
            Y_hat = [1 if i > 0.5 else 0 for i in probabilities]
            return np.array(Y_hat)



# MAIN
ngrams = (1, 2)
training_portion = 1.0
languages = ['English', 'French', 'German']
# Load dataset from https://www.kaggle.com/basilb2s/language-detection
# I uploaded it to my git repo:
if path.exists("Language_Detection.csv"):
    url = "Language_Detection.csv"
else:
    url = ("https://raw.githubusercontent.com/"
                            "iarroyof/ukp_app/main/Language%20Detection.csv")
print("Loading data from: {}".format(url))
# Prepare train and test data for three languages. Each dcoument is represented
# first as a list of character n-gram tokens. Then the documents are encoded
# in binary vectors. You can test the code for any subset of the available 
# languages in the dataset
X_train, Y_train, X_test, Y_test, vocabulary = prepare_data(
                            url, sample=training_portion, languages=languages)

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
Y_hat = ovr_clf.predict(X_test)

print("\nSample of predictions on the test set:\n")
result = pd.DataFrame({"Y": [label2class[label] for label in Yl_test],
                        "Y_hat": [label2class[label] for label in Y_hat]})
print(result.head(50))

# Print test performace results
print("\nTest Accuracy: {}".format(accuracy(Yl_test, Y_hat)))
print("If you want to detect language from an input document"
        " please give it as a *.txt. Otherwise only press ENTER"
        " to finish:\n>> ")
filename = input()

while filename.endswith(".txt"):
    try:
        f = open(filename)
        try:
            test_doc = " ".join(f.readlines())
        except UnicodeError:
            f = open(filename, encoding="latin-1")
            test_doc = " ".join(f.readlines())
        Xtest, _ = categorical_encoder([test_doc], vocabulary=vocabulary)
        Xtest = Xtest.toarray()
        Y_hat = ovr_clf.predict(Xtest)
        print(Y_hat)
        print("Predicted lnaguage: {}".format(label2class[Y_hat[0]]))
        print("Another file?..")
        filename = input()
    except:
        print("File not found. Try again...")
        filename = input()
