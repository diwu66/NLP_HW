# The following Data processing based on https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html
""" 1. Usage """
import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint
from sklearn import metrics   # --> metrics.f1_score(target, pred, average = "macro")
from scipy.sparse import csr_matrix, hstack


class Perceptron(object):
    def __init__(self):
        self.eta = 1     # eat: learning_rate
        self.max_iter = 10000   # max number of iter --> stop

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
            x = dt[i, :].todense()   # x is a np.matrix (1, 129792)
            y = labels[i]
            # compute "phi(x, y)" function
            phi_xy = np.zeros(n * self.K)
            phi_xy[(y*n):((y+1)*n), ] = x

            # for all other yhat (not equal to yi) find y_HAT that maximize [[ W_transpose %*% phi(x, yhat) ]]
            #input, for each sample with feature_array x
            phi_xy_tile = np.array( np.tile(x, self.K) )
            w_m = np.asarray(self.w)
            all_result = phi_xy_tile * w_m
            # split array with NK*1 into K part --> then Calculate the Sum of N --> argmax
            all_result_split = np.array_split(all_result, self.K)
            phi_all = [all_result_split[j].sum() for j in range(self.K)]
            phi_all[y] = -1  # let the true block y = -1 --> argmax of others..
            yhat = np.argmax(phi_all)

            phi_xyhat = np.zeros(n * self.K)
            phi_xyhat[(yhat*n):((yhat+1)*n), ] = x
            # update W <-- W + eta * [phi(xi,yi) - phi(xi, y_HAT)]
            self.w = self.w + self.eta * (phi_xy - phi_xyhat)

            if iteration%200 == 0:
                print(iteration)

            iteration = iteration + 1

    def pred_(self, x):
        ## input, for each sample with feature_array x
        phi_xy = np.array( np.tile(x, self.K) )
        w_m = np.asarray(self.w)
        all_result = phi_xy * w_m

        # split array with NK*1 into K part --> then Calculate the Sum of N --> argmax
        all_result_split = np.array_split(all_result, self.K)
        ypred = np.argmax([all_result_split[j].sum() for j in range(self.K)])
        return ypred

    def predict(self, features_test):
        mat_one = csr_matrix( np.ones(features_test.get_shape()[0]) )
        dt = hstack((features_test, mat_one.transpose()))   # add 1 for multiplying intercept b
        dt = dt.tocsr()

        labels_pred = []
        for f in range(features_test.get_shape()[0]):
            x = dt[f, :].todense()   # x is a np.matrix (1, ???)
            labels_pred.append(self.pred_(x))
        return labels_pred


""" MAIN FUNCTION """
""" 1. Read data """
newsgroups_train = fetch_20newsgroups(subset='train')

""" 2. Converting text to vectors """
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features = vectorizer.fit_transform(newsgroups_train.data)


""" 3. Training using the Perceptron Algorithm """
labels = newsgroups_train.target
# train
P = Perceptron()
P.train(features, labels)
train_pred = P.predict(features)
print( metrics.confusion_matrix(labels, train_pred))

# predict
newsgroups_test = fetch_20newsgroups(subset='test')
test_features = vectorizer.transform(newsgroups_test.data)
test_labels = newsgroups_test.target

test_pred = P.predict(test_features)

# measure:
print( metrics.confusion_matrix(test_labels, test_pred) )

