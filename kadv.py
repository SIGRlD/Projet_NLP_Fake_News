import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

class K_Avocats():
    def __init__(self, num_classes, epsilon=0.1, seuil=0.9, k_voisins=4, n_iterations=25):
        self.k = num_classes
        self.e = epsilon
        self.s = seuil
        self.k_voisins = k_voisins
        self.n = n_iterations

    def fit(self, X, y):
        n = X.shape[0]
        self.X = X
        self.y = np.array(y, dtype=int)
        self.poids = np.full((n, self.k), 0.5)

        # Initialisation centroïdes : moyenne des coordonnées des points de chaque classe
        self.centroïdes = np.zeros((self.k, X.shape[1]))
        for c in range(self.k):
            self.centroïdes[c] = X[self.y == c].mean(axis=0)

        accuracy = 0
        it = 0

        while accuracy < self.s or it < self.n:
            print(f"\nIteration {it}")
            # Étape 1 : clustering pondéré
            clusters = self._clustering_pondere()

            # Étape 2 : ajustement poids (phase d’orientation)
            self._renforcer_clusters(clusters)

            # Étape 3 : prédiction via KNN pondéré
            y_pred = self._predict_classes()

            if it > 2:
                self._presentation_preuve()
                self._debats()

            # Étape 4 : objections (poids ciblés)
            self._objection(y_pred, clusters)


            y_pred = self._predict_classes()
            # Étape 5 : mesure accuracy
            accuracy = accuracy_score(self.y, y_pred)
            print("Accuracy:", accuracy)

            it += 1

    def _clustering_pondere(self):
        clusters = []
        for i, x in enumerate(self.X):
            dists = [np.linalg.norm(x - c) * (1 - self.poids[i, j]) for j, c in enumerate(self.centroïdes)]
            clusters.append(np.argmin(dists))
        return np.array(clusters)

    def _renforcer_clusters(self, clusters):
        for i, cluster in enumerate(clusters):
            self.poids[i, cluster] = min(1.0, self.poids[i, cluster] + self.e / 2)

    def _predict_classes(self):
        # Utilise les k plus proches voisins pour prédire la classe
        voisins = NearestNeighbors(n_neighbors=self.k_voisins).fit(self.X)
        _, indices = voisins.kneighbors(self.X)

        y_pred = []
        for i, idx in enumerate(indices):
            score = np.zeros(self.k)
            for j in idx:
                if i == j: continue  # on peut exclure soi-même si besoin
                dist = np.linalg.norm(self.X[i] - self.X[j]) + 1e-6
                poids = 1 / (dist + 1)
                score += poids * self.poids[j]
            y_pred.append(np.argmax(score))
        return np.array(y_pred)

    def _objection(self, y_pred, clusters):
        for i in range(len(self.X)):
            vrai = int(self.y[i])
            pred = int(y_pred[i])
            if vrai != pred:
                # Objection principale
                self.poids[i, pred] = max(0.0, self.poids[i, pred] - self.e / 2)
                self.poids[i, vrai] = min(1.0, self.poids[i, vrai] + self.e)

                # Renforcement diffus dans le cluster voisin
                voisins = NearestNeighbors(n_neighbors=self.k_voisins).fit(self.X)
                _, indices = voisins.kneighbors([self.X[i]])
                for j in indices[0]:
                    if j == i: continue
                    classe_cluster = clusters[j]
                    if self.poids[j, vrai] < self.poids[j, classe_cluster]:
                        self.poids[j, vrai] = min(self.poids[j, classe_cluster], self.poids[j, vrai] + self.e / 5)

    def _presentation_preuve(self):
        for i in range(len(self.X)):
            vrai = self.y[i]
            dist = np.linalg.norm(self.X[i] - self.centroïdes[vrai])
            seuil_preuve = np.percentile([np.linalg.norm(x - self.centroïdes[self.y[j]]) for j, x in enumerate(self.X)], 90)

            if dist > seuil_preuve:
                # Renforcement de la bonne classe
                self.poids[i, vrai] = min(1.0, self.poids[i, vrai] + self.e)

                # Réduction des poids des autres classes
                for j in range(self.k):
                    if j != vrai:
                        self.poids[i, j] = max(0.0, self.poids[i, j] - self.e / 2)

    def _debats(self):
        for i in range(len(self.X)):
            p = self.poids[i]
            diffs = np.abs(p[:, None] - p[None, :])
            similaires = (diffs < 0.1).sum(axis=1)  # nombre de poids proches pour chaque classe

            count_similaires = (similaires > 1).sum()  # classes proches entre elles

            if count_similaires >= 4:
                self.poids[i] *= (1 - 2 * self.e)
            elif count_similaires == 3:
                self.poids[i] *= (1 - self.e)
            elif count_similaires == 2:
                self.poids[i] *= (1 - self.e / 2)

            # Normalisation facultative
            self.poids[i] = np.clip(self.poids[i], 0.0, 1.0)

    def predict(self, X_new, return_scores=False, seuil_diff=0.05):
        voisins = NearestNeighbors(n_neighbors=self.k_voisins).fit(self.X)
        _, indices = voisins.kneighbors(X_new)

        y_pred = []
        for i, idx in enumerate(indices):
            score = np.zeros(self.k)
            for j in idx:
                if i == j: continue  # on peut exclure soi-même si besoin
                dist = np.linalg.norm(self.X[i] - self.X[j]) + 1e-6
                poids = 1 / (dist + 1)
                score += poids * self.poids[j]
            y_pred.append(np.argmax(score))
        return np.array(y_pred)


