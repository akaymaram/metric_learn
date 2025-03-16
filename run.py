import numpy as np
from metric_learn import LMNN, NCA, LFDA
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

iris_data = load_iris()
X = iris_data['data']
Y = iris_data['target']


# Large Margin Nearest Neighbor Metric Learning (LMNN)
lmnn = LMNN(n_neighbors=5, learn_rate=1e-6)
lmnn.fit(X, Y)

# Neighborhood Components Analysis (NCA)
nca = NCA(max_iter=1000)
nca.fit(X, Y)

# Local Fisher Discriminant Analysis (LFDA)
lfda = LFDA(k=2, n_components=2)
lfda.fit(X, Y)
for model in [lmnn, nca, lfda]:
	clf = make_pipeline(model, KNeighborsClassifier())
	print(cross_val_score(clf, X, Y))


