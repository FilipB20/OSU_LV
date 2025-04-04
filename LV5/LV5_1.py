import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

#a)
plt.figure(figsize=(8, 6))

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', label='Train data')

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', label='Test data')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Umjetni binarni klasifikacijski problem')
plt.legend()
plt.grid(True)
plt.show()

#b)
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

#c)
coef = LogRegression_model.coef_
intercept = LogRegression_model.intercept_
print(coef)
print(intercept)
x1_values = np.array([X_train[:, 0].min(), X_train[:, 0].max()])
x2_values = -(intercept + coef[0, 0] * x1_values) / coef[0, 1]

plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k', label='Podaci za učenje')
plt.plot(x1_values, x2_values, c='red', label='Granica odluke')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Granica odluke naučenog modela logističke regresije')
plt.legend()
plt.grid(True)
plt.show()

#d)
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

y_pred = LogRegression_model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrica zabune:")
print(conf_matrix)

accuracy = accuracy_score(y_test, y_pred)
print("Točnost: {:.2f}".format(accuracy))

precision = precision_score(y_test, y_pred)
print("Preciznost: {:.2f}".format(precision))

recall = recall_score(y_test, y_pred)
print("Odziv: {:.2f}".format(recall))

#e)
y_pred = LogRegression_model.predict(X_test)

correct_indices = np.where(y_pred == y_test)[0]
incorrect_indices = np.where(y_pred != y_test)[0]

plt.figure(figsize=(8, 6))
plt.scatter(X_test[correct_indices, 0], X_test[correct_indices, 1], c='green', label='Dobro klasificirani')
plt.scatter(X_test[incorrect_indices, 0], X_test[incorrect_indices, 1], c='black', label='Pogrešno klasificirani')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Skup za testiranje s označenim dobro i pogrešno klasificiranim primjerima')
plt.legend()
plt.grid(True)
plt.show()
