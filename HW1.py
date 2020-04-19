import numpy as np
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class Perceptron:
    def __init__(self, N, K, lr=0.01):  # N: # of features; K: # of labels; lr: learning rate
        self.w = np.zeros([N, K])
        self.lr = lr
        # self.w_mean = self.w.copy()

    def train(self, data, labels, max_epoch=100):
        m = list(range(data.shape[0]))

        for i in range(max_epoch):
            random.shuffle(m)

            err = 0
            for j in m:
                x = data[j,:]
                y = labels[j]
                y_hat = np.argmax(x @ self.w)

                if y_hat == y:
                    pass
                else:
                    err += 1
                    self.w[:, y] += self.lr*x
                    self.w[:, y_hat] -= self.lr*x
                # self.w_mean += self.w

            # create decreasing learning rate...
            self.lr = self.lr*0.95
            
            if i % 20 == 0:
                print('Epoch {} Training Error {} '.format(i, err/len(m)))

    def pred(self,  data):
        y_hat = data @ self.w
        y_pred = np.argmax(y_hat, axis=1)
        return y_pred


## Train set
newsgroups_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))
X_train, y_train = newsgroups_train['data'], newsgroups_train['target']
ntrain = len(X_train)

## Test set
newsgroups_test = fetch_20newsgroups(subset='test',shuffle=True, random_state=42,remove=('headers', 'footers', 'quotes'))
X_test, y_test = newsgroups_test['data'], newsgroups_test['target']

## Feature Selection: dimension reduction
X = X_train+X_test
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
features = vectorizer.fit_transform(X)
svd = TruncatedSVD(n_components=800, n_iter=2000, algorithm='arpack')  # feature size: 800
X = svd.fit_transform(features)

X_train = X[:ntrain]
X_test = X[ntrain:]

## Train 
max_epoch=1000
model = Perceptron(N=X_train.shape[1], K=max(y_train)+1, lr = 0.01)
model.train(X_train, y_train,  max_epoch )

## Predict
pred = model.pred( X_test )
print( sum(pred == y_test)/len(y_test) )
