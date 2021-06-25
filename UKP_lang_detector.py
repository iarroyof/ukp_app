from collections import deque
import numpy as np
import pandas as pd
 
def tokenize(doc, ngram_range=(1, 3)):
 
    collected_ngrams = []
    for ng_size in range(*ngram_range):
        window = deque(maxlen=ng_size)
        for ch in doc:
            window.append(ch)
            collected_ngrams.append(''.join(list(window)))
 
    return collected_ngrams


def accuracy(y, y_hat):
    eqs = 0
    for a, b in zip(y, y_hat):
        if a == b:
            eqs += 1
 
    return eqs / len(y)


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
    test_data = dataset.iloc[int(0.7 * len(dataset.index)):]
 
    X_train_data = train_data.Text.apply(tokenize)
    X_train_data = X_train_data.apply(set)
    X_train_data = X_train_data.apply(list)
    Y_train_data = train_data.Language
 
    Xtrain = []
    Ytrain = []
 
    for x, y in zip(X_train_data, Y_train_data):
        Ytrain += [y] * len(x)
        Xtrain += x
 
    X_test_data = test_data.Text.apply(tokenize)
    X_test_data = X_test_data.apply(set)
    X_test_data = X_test_data.apply(list)
    Y_test_data = test_data.Language
 
    Ytest = []
    Xtest = []
    for x, y in zip(X_test_data, Y_test_data):
        Ytest += [y] * len(x)
        Xtest += x
 
    return Xtrain, Ytrain, Xtest, Ytest


class BayesClassifier:
    """This class implements a Naïve Bayes classifier specifically coded for
    two discrete RVs, X and Y. X is the RV taking on n-grams of characters 
    composing input documents, and Y is the RV taking on languages. Thus to
    infer the language an input document is written in, I compute:

    P(Y=y|X=x) = P(Y=y) \Pi_{w\in x} P(Y=y|w)

    Parameters:
    ngram_range: range of sizes of the ngrams of characters an input document
                 splits in. 
    exact_estimator: whether the likelihood is computed using partition function
                     or using only counts. 
    """
    def __init__(self, ngram_range=(1, 3), out_distributions=False, exact_estimator=False):
        self.ngram_range = ngram_range
        self.out_dist = out_distributions
        self.exact = exact_estimator


    def tokenize(self, doc):

        collected_ngrams = []
        for ng_size in range(*self.ngram_range):
            window = deque(maxlen=ng_size)
            for ch in doc:
                window.append(ch)
                collected_ngrams.append(''.join(list(window)))

        return collected_ngrams


    def fit(self, X, Y):
        """Bayes training
        I first create a contingecy table
        Each cell contains the result of the indicator product function 
        f(x, y) = 1 if x == x' and y == y' ? 0 otherwise.

        I create sample spaces for each RV"""
        (self.omega_x, Tx) = np.unique(X, return_counts=True)
        (self.omega_y, Ty) = np.unique(Y, return_counts=True)
        self.PY = {y: ty for y, ty in zip(self.omega_y, Ty)} 

        # Create contigency table (Kronecker product)
        f_xy = {}
        for x in self.omega_x:
            for y in self.omega_y:
                f_xy[(x, y)] = sum([int(x_ == x and y_ == y)
                    for x_, y_ in zip(X, Y)])


         # Posterior computations
        if self.exact:
            self.PYgX = {}
            for y in self.omega_y:
                for x in self.omega_x:
                    Zx = sum([f_xy[(x, y_)] for y_ in self.omega_y])
                    self.PYgX[(y, x)] = f_xy[(x, y)] / Zx
        else:
            self.PYgX = f_xy

        return self


    def posterior(self, text):
        """ This function uses Naïve Assumption to compute
            posterior distribution of an input document. 
        """
        tokens = list(set(self.tokenize(text)))
        prod = 1.0 # [1.0]  *  len(self.omega_y)
        pmfs = [] 
        for y in self.omega_y:
            for x in tokens:
                try:
                    prod += np.log2(self.PYgX[(y, x)])
                except KeyError:
                    pass
            pmfs.append(self.PY[y] * prod)

        return self.omega_y[np.argmax(pmfs)] 
        #for ygx in pmfs.items():
        #    prod = np.multiply(prod, ygx)
        #if self.out_dist:
        #    return list(zip(self.omega_y, prod))
        #else:
        #    return self.omega_y[np.argmax(prod)]


    def predict(self, X):
        predictions = [self.posterior(x) for x in X]
        return predictions

#MAIN
url = "https://raw.githubusercontent.com/iarroyof/ukp_app/main/Language%20Detection.csv"
X_train, Y_train, X_test, Y_test = prepare_data_RVs(url, sample=0.01, languages=['English', 'German', 'Spanish'])
 
bayes = BayesClassifier(ngram_range=(1, 4), exact_estimator=True)

bayes.fit(X_train, Y_train)

Y_hat = bayes.predict(X_test)

accuracy(Y_test, Y_hat)
