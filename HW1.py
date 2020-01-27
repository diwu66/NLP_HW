# The following Data processing based on https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
""" 1. Usage """
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics   # --> metrics.f1_score(target, pred, average = "macro")
from scipy.sparse import csr_matrix, hstack


class Perceptron(object):
    def __init__(self):
        self.eta = 0.0005   # eat: learning_rate
        self.max_iter = 45000   # max number of iter --> stop (about 4 epochs..)

    def train(self, features, labels):
        dD = features.get_shape()[0]  # dD: number of samples
        # initialize list w = (w, b), the last "1" is for b-intercept
        n = features.get_shape()[1] + 1   # n: number of features/columns with the intercept b
        self.K = np.unique(labels).__len__()   # K: unique number of labels
        self.w = [0.0] * n * self.K   # w._len_ = #of(features) * #of(classes) = 129792 * 20...

        # initialize correct count
        iteration = 0

        mat_one = csr_matrix(np.ones(dD))
        dt = hstack((features, mat_one.transpose()))   # add 1 for multiplying intercept b
        dt = dt.tocsr()

        while iteration < self.max_iter:
            # random sample i -->dD sample, (xi, yi)
            i = random.randint(0, dD-1)
            x = np.array( dt[i, :].todense()).flatten()   # x is a np.array( 129792, 1 )
            y = labels[i]

            row = np.arange(0, n*self.K, dtype=int)
            col = np.arange(0, self.K, dtype=int).repeat(n)
            data = np.tile(np.array(x), self.K)
            Phi_mat = csr_matrix( (data,(row,col)), shape=(n*self.K, self.K) ).todense()

            wPhi = self.w * Phi_mat

            yhat = wPhi.argmax()
            phi_y_yhat = np.zeros(n * self.K)
            phi_y_yhat[(y*n):((y+1)*n), ] = x
            phi_y_yhat[(yhat*n):((yhat+1)*n), ] = -x

            # update self.w
            self.w = self.w + self.eta * phi_y_yhat

            if iteration%200 == 0:
                print(iteration)

            iteration = iteration + 1

    def pred_(self, x, n):
        ## input, for each sample with feature_array x
        row = np.arange(0, n*self.K, dtype=int)
        col = np.arange(0, self.K, dtype=int).repeat(n)
        data = np.tile(np.array(x), self.K).flatten()
        Phi_mat = csr_matrix( (data,(row,col)), shape=(n*self.K, self.K) ).todense()

        wPhi = self.w * Phi_mat
        ypred = wPhi.argmax()
        return ypred

    def predict(self, features_test):
        mat_one = csr_matrix( np.ones(features_test.get_shape()[0]) )
        n_test = features_test.get_shape()[1] + 1
        dt = hstack((features_test, mat_one.transpose()))   # add 1 for multiplying intercept b
        dt = dt.tocsr()

        labels_pred = []
        for f in range(features_test.get_shape()[0]):
            x = dt[f, :].todense()   # x is a np.array
            labels_pred.append(self.pred_(x, n_test))
            if f%200 == 0:
                print(f)

        return labels_pred


""" MAIN FUNCTION 
  1. Read data
"""
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))

""" 2. Converting text to vectors 
 One first need to turn the text into vectors of numerical values suitable for statistical analysis. This can be achieved with the utilities
of the sklearn.feature_extraction.text as demonstrated in the following example that extract TF-IDF vectors of unigram tokens from a subset of 20news: 
"""
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features = vectorizer.fit_transform(newsgroups_train.data)


### 3. Training using the Perceptron Algorithm
labels = newsgroups_train.target

# train
P = Perceptron()
P.train(features, labels)


# predict
newsgroups_test = fetch_20newsgroups(subset='test',shuffle=True, random_state=42,remove=('headers', 'footers', 'quotes'))
test_features = vectorizer.transform(newsgroups_test.data)
test_labels = newsgroups_test.target

# pred
test_pred = P.predict(test_features)
confusion_mat = metrics.confusion_matrix(test_labels, test_pred)
accuracy = metrics.accuracy_score(test_labels, test_pred)
print(confusion_mat)
"""
[[101   0   2   0   1   2   0   0   0   0   0   2   2   1 191   5   1   0    2   9]
 [  8  17  16   0   3  17   2   0   0   1   1   1  12   2 302   2   1   2  0   2]
 [ 14   0 134   1   3   6   1   0   0   1   3   3   2   0 222   0   2   2  0   0]
 [ 18   0  28  12   7  11  10   0   0   4   1   1  15   2 276   2   2   2  1   0]
 [ 10   0   2   1  68   4  10   0   0   0   1   1   7   4 270   3   2   1  0   1]
 [ 10   3  14   0   1 149   2   0   0   0   1   1   5   2 203   1   0   0  0   3]
 [  8   0   1   0   9   5 173   0   0   1   0   2   3   1 184   0   1   0  1   1]
 [ 35   0   7   0   2   5   9   2   1   2   4   2  12   4 298   1   4   3  3   2]
 [ 16   0   7   0   4   5   8   0  20   4   2   1   6   3 312   2   3   0  2   3]
 [ 10   0   1   0   1   3   1   0   0 157   6   1   0   0 211   3   1   0  1   1]
 [  6   0   2   0   1   2   3   0   0   4 168   0   1   1 207   1   1   0  1   1]
 [ 11   0   8   0   3   4   0   0   0   1   1  67   4   1 291   0   2   0  1   2]
 [ 21   0   4   0   5   5   6   0   0   2   0   4  73   2 264   0   3   3  0   1]
 [  3   0   0   0   0   0   2   0   0   1   1   0   5 121 256   2   2   1  1   1]
 [  3   0   2   0   0   1   4   0   0   0   0   1   2   2 378   0   0   0  1   0]
 [ 21   0   1   0   1   4   1   0   0   2   1   3   2   1 253  98   0   0  2   8]
 [ 18   0   4   0   0   3   4   0   0   1   0   1   0   1 231   1  89   2  6   3]
 [ 20   0   2   0   2   2   2   0   0   2   3   1   2   1 255   3   1  75  5   0]
 [ 17   0   1   0   1   6   0   0   0   1   0   1   2   5 198   0  39   0 37   2]
 [ 27   0   2   0   1   2   0   0   0   0   2   1   1   1 170   8   7   1  0  28]]
"""

print(accuracy)
# 0.26

