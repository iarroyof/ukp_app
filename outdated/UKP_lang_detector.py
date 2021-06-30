# The Bayes Theorem was first proposed for Language Identification in the next references:
# Dunning, T. (1994). Statistical identification of language (pp. 94-273). Las Cruces, NM, USA: Computing Research Laboratory, New Mexico State University.
# Cavnar, W. B., & Trenkle, J. M. (1994, April). N-gram-based text categorization. In Proceedings of SDAIR-94, 3rd annual symposium on document analysis and information retrieval (Vol. 161175).
# Why Bayes?
# Here is the basic version of this approach to show my general modeling idea of the problem and the details of its implementation.


from collections import deque
import numpy as np
import pandas as pd
import unicodedata
import string




def filter_non_printable(str):
    printable = {'Lu', 'Ll'}
  
    return ''.join(c for c in str if unicodedata.category(c) in printable or c == ' ')


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
    Xtrain = train_data.Text
    Ytrain = train_data.Language

    test_data = dataset.iloc[int(0.7 * len(dataset.index)):]
    Xtest = test_data.Text
    Ytest = test_data.Language

    return Xtrain, Ytrain, Xtest, Ytest


class BayesClassifier:
    """This class implements a Naïve Bayes classifier specifically coded for
    two discrete RVs, X and Y. X is the RV taking on n-grams of characters 
    composing input documents, and Y is the RV taking on languages. Thus to
    infer the language an input document is written in, I compute:

    $$P(Y=y|X=x) = P(Y=y) \Pi_{w\in x} P(Y=y|w)$$

    Parameters:
    -----------

    ngram_range: range of sizes of the ngrams of characters an input document
                 splits in.
    exact_estimator: whether the likelihood is computed using partition function
                     or using only counts.
    """
    def __init__(self, ngram_range=(1, 3), exact_estimator=False):
        self.ngram_range = ngram_range
        self.exact = exact_estimator


    def tokenize(self, doc):

        collected_ngrams = []
        doc = doc.translate(str.maketrans('', '', string.punctuation))
        doc = filter_non_printable(doc)
        for ng_size in range(*self.ngram_range):
            window = deque(maxlen=ng_size)
            for ch in doc:
                window.append(ch)
                collected_ngrams.append(''.join(list(window)))

        return collected_ngrams


    def fill_RVs(self, X_data, Y_data=None):
        X_data = X_data.apply(self.tokenize)
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


    def fit(self, X, Y):
        """Bayes training
        I first create a contingecy table
        Each cell contains the area of the indicator product function
        f(x, y) = 1 if x == x' and y == y' ? 0 otherwise.
        """
        # Create sample spaces and compute class probability distribution P(Y)
        X, Y = self.fill_RVs(X, Y)
        (self.omega_x, Tx) = np.unique(X, return_counts=True)
        (self.omega_y, Ty) = np.unique(Y, return_counts=True)
        self.PY = {y: ty / sum(Ty) for y, ty in zip(self.omega_y, Ty)}
        #self.PX = {x: tx / sum(Tx) for x, tx in zip(self.omega_x, Tx)} 

        # Create contigency table
        f_xy = {}
        for x, y in zip(X, Y):
            if (x, y) in f_xy:
                f_xy[(x, y)] += 1.0
            else:
                f_xy[(x, y)] = 1.0

        # Posterior computations
        self.PYgX = {}
        if self.exact:
            for y in self.omega_y:
                for x in self.omega_x:
                    Zx = []
                    for y_ in self.omega_y:
                        try:
                            Zx.append(f_xy[(x, y_)])
                        except KeyError:
                            pass
                    try:
                        self.PYgX[(y, x)] = f_xy[(x, y)] / sum(Zx)
                    except KeyError:
                        self.PYgX[(y, x)] = 0.0
        else:
            for k, v in f_xy.items():
                self.PYgX[(k[1], k[0])] = v / Tx[np.where(self.omega_x==k[0])]

        return self


    def posterior(self, text):
        """ This function uses Naïve Assumption to compute
            posterior distribution of an input document.
        """
        tokens = list(set(self.tokenize(text)))

        for y in self.omega_y:
            for x in tokens:
                try:
                    self.PY[y] *= self.PYgX[(y, x)]
                    #if p > 0.0:
                    #    prod += np.log2(p)
                    #else:
                    #    prod += 0.0
                except KeyError:
                    pass

        return max(self.PY, key=self.PY.get)


    def predict(self, X):
        predictions = [self.posterior(x) for x in X]

        return predictions


# MAIN:
exact = False
ngrams = (1, 4)
training_portion = 1.0
languages = ['English', 'Spanish']
url = "https://raw.githubusercontent.com/iarroyof/ukp_app/main/Language%20Detection.csv"

# Load train and test data for three languages:
X_train, Y_train, X_test, Y_test = prepare_data_RVs(url, sample=training_portion, languages=languages)

bayes = BayesClassifier(ngram_range=ngrams, exact_estimator=exact)

print("Training the classifier with {}% of the input data for {} languages: {}... ***more training data needs more training time***".format(training_portion * 100, len(languages), ", ".join(languages)))
bayes.fit(X_train, Y_train)

Y_hat = bayes.predict(X_test)
print(pd.DataFrame({"Y": Y_test, "Y_hat": Y_hat}).head(50))
print("Test accuracy of the implementation:")
print(accuracy(Y_test, Y_hat))

print("Now write an input sentence or plain text file name (*.txt) in one of {} languages: {}".format(len(languages), ", ".join(languages)))

test_string = input()
if test_string.endswith(".txt"):
    with open(test_string, encoding="utf8", errors="ignore") as f:
        test_string = " ".join(f.readlines())

Y_hat = bayes.predict([test_string])

print("Predicted language: {}".format(Y_hat))
