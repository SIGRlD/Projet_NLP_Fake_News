from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import numpy as np


class classifier_models():
    def __init__(self, embedding, labels,
                 method=LogisticRegression(C=5, max_iter=10000, class_weight='balanced'), rebalance=False):
        """
        :param embedding: liste des embedding
        :param method: module sklearn
        """
        labels = np.array(labels)
        if rebalance:
            if hasattr(embedding, "toarray"):
                embedding = embedding.toarray()
            X_pos = embedding[labels == 1]
            X_neg = embedding[labels == 0]
            print(X_neg)
            # downsample les 0
            X_neg_down, y_neg_down = resample(X_neg, labels[labels == 0],
                                              replace=False,
                                              n_samples=X_pos.shape[0],
                                              random_state=42)

            embedding = np.vstack([X_pos, X_neg_down])
            labels = np.array([1] * X_pos.shape[0] + [0] * y_neg_down.shape[0])
        if(embedding.shape[1] > 50):
            svd = TruncatedSVD(n_components=50)  # essaie 50, 100, 200
            self.X = svd.fit_transform(embedding)
            self.svd = svd
        else:
            self.X = embedding
            self.svd = None
        self.algo = method
        self.algo.fit(self.X, labels)

    def predict_labels(self, X):
        """
        :param X: entrées
        :return: labels pour chaque entrée
        """
        if X.shape[1] != self.X.shape[1]:
            if self.svd is None:
                print("Dimensions de l'entrée incohérentes. Attendu: ", self.X.shape[1], ", obtenu: ", X.shape[1])
                return 1
            else:
                X = self.svd.transform(X)

        return self.algo.predict(X)

    def predict_scores(self, X):
        """
        :param X: entrées
        :return: scores pour chaque label
        """
        if X.shape[1] != self.X.shape[1]:
            if self.svd is None:
                print("Dimensions de l'entrée incohérentes. Attendu: ", self.X.shape[1], ", obtenu: ", X.shape[1])
                return 1
            else:
                X = self.svd.transform(X)

        return self.algo.predict_proba(X)[:, 1]

