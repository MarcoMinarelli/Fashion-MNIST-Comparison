from sklearn import metrics
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

GINI = 'gini'
ENTROPY = 'entropy'

def report(target, pred, clfName):
    print(clfName)
    print("Multiclass accuracy: %.2f%%" %
          (100 * metrics.accuracy_score(target, pred)))
    cm = metrics.confusion_matrix(target, pred)
    print('Contingency matrix: (each row: examples of a given true class)')
    print(cm)
    print('Classification report')
    print(metrics.classification_report(target, pred))


def Bernoulli(X_train, y_train, X_test, y_test):
    classifier = naive_bayes.BernoulliNB(alpha=1, binarize=0.01,
                                         fit_prior=True)
    # binarize = 0.01 per sfruttare tutto ciò che non è zero: i pixel "di background" diventano 0 e i pixel di
    # foreground diventano 1
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    report(y_test, y_pred, "Bernoulli Naive Bayes")


def Multinomial(X_train, y_train, X_test, y_test):
    classifier = naive_bayes.MultinomialNB(alpha=1, fit_prior=True)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    report(y_test, y_pred, "Multinomial Naive Bayes")


def DecisionTree(X_train, y_train, X_test, y_test, crit = GINI):
    classifier = DecisionTreeClassifier(criterion=crit)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    report(y_test, y_pred, "Decision Tree with " + crit)


def RandomForests(X_train, y_train, X_test, y_test, numTrees=100):
    classifier = RandomForestClassifier(numTrees)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    report(y_test, y_pred, "Random Forests with " + str(numTrees) + " trees")


def load_mnist(path, kind='train'):
    import os
    import gzip

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


if __name__ == '__main__':
    X_train, y_train = load_mnist('data/')
    X_test, y_test = load_mnist('data/', kind='t10k')
    Bernoulli(X_train, y_train, X_test, y_test)
    Multinomial(X_train, y_train, X_test, y_test)
    DecisionTree(X_train, y_train, X_test, y_test)
    DecisionTree(X_train, y_train, X_test, y_test, ENTROPY)
    RandomForests(X_train, y_train, X_test, y_test)
    RandomForests(X_train, y_train, X_test, y_test, 500)
    RandomForests(X_train, y_train, X_test, y_test, 1000)
