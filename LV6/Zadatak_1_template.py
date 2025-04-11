import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

# Zadatak 6.5.1
"""Skripta zadatak_1.py ucitava Social_Network_Ads.csv skup podataka [2].
 Ovaj skup sadrži podatke o korisnicima koji jesu ili nisu napravili kupovinu za prikazani oglas.
 Podaci o korisnicima su spol, dob i procijenjena placa. Razmatra se binarni klasifikacijski
 problem gdje su dob i procijenjena placa ulazne velicine, dok je kupovina (0 ili 1) izlazna
 velicina. Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna funkcija
 plot_decision_region [1]. Podaci su podijeljeni na skup za ucenje i skup za testiranje modela
 u omjeru 80%-20% te su standardizirani. Izgraden je model logisticke regresije te je izracunata
 njegova tocnost na skupu podataka za ucenje i skupu podataka za testiranje. Potrebno je:
 1. Izradite algoritam KNN na skupu podataka za ucenje (uz K=5). Izracunajte tocnost
 klasifikacije na skupu podataka za ucenje i skupu podataka za testiranje. Usporedite
 dobivene rezultate s rezultatima logistiˇ cke regresije. Što primjecujete vezano uz dobivenu
 granicu odluke KNN modela?
 2. Kako izgleda granica odluke kada je K =1 i kada je K = 100?"""

KNN_model = KNeighborsClassifier(n_neighbors = 5)
KNN_model.fit(X_train_n, y_train)

y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict(X_test_n)

print("KNN (K=5): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()
plt.show()

"""KNN model daje bolje rezulate na danim podacima u odnosu na model logističke regresije
jer stvara nelinearnu granicu odluke što je ključno kada podaci nisu linearno odvojivi."""

KNN_model_K1 = KNeighborsClassifier(n_neighbors = 1)
KNN_model_K1.fit(X_train_n, y_train)

y_train_p_KNN_K1 = KNN_model_K1.predict(X_train_n)
y_test_p_KNN_K1 = KNN_model_K1.predict(X_test_n)

print("KNN (K=1): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN_K1))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN_K1))))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model_K1)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN_K1))))
plt.tight_layout()
plt.show()

KNN_model_K100 = KNeighborsClassifier(n_neighbors = 100)
KNN_model_K100.fit(X_train_n, y_train)

y_train_p_KNN_K100 = KNN_model_K100.predict(X_train_n)
y_test_p_KNN_K100 = KNN_model_K100.predict(X_test_n)

print("KNN (K=100): ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN_K100))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN_K100))))

plot_decision_regions(X_train_n, y_train, classifier=KNN_model_K100)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN_K100))))
plt.tight_layout()
plt.show()

"""Za slučaj kada je K=1 možemo reći da se radi o overfitting-u jer je točnost na podacima za treniranje 
visoka, ali na podacima za testiranje nezadovoljavajuća. Za K=100 se radi o underfitting-u jer model ima nezadovoljavajuće 
performanse i na skupu za učenje i na skupu za testiranje."""

# Zadatak 6.5.2 
""" Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra K
 algoritma KNN za podatke iz Zadatka 1."""

from sklearn.model_selection import cross_val_score
model = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(KNN_model, X_train, y_train, cv=5)
print("Rezultati iteracija unakrsne validacije: ", scores)

param_grid = {'n_neighbors': np.arange(1, 21)}  
grid_search = GridSearchCV(KNN_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_n, y_train)

best_knn_model = grid_search.best_estimator_
test_accuracy = accuracy_score(y_test, best_knn_model.predict(X_test_n))
print(grid_search.best_params_)
print(grid_search.best_score_)
print(f"Najbolji model KNN (K={grid_search.best_params_['n_neighbors']}) postiže točnost od {test_accuracy:.3f} na skupu za testiranje.")

# Zadatak 6.5.3
"""Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkciju
 te prikažite dobivenu granicu odluke. Mijenjajte vrijednost hiperparametra C i γ. Kako promjena
 ovih hiperparametara utjece na granicu odluke te pogrešku na skupu podataka za testiranje?
 Mijenjajte tip kernela koji se koristi. Što primjecujete?"""

SVM_model = svm.SVC(kernel='rbf', gamma = 1, C=1)
SVM_model.fit(X_train_n, y_train)

y_train_p_SVM = SVM_model.predict(X_train_n)
y_test_p_SVM = SVM_model.predict(X_test_n)
print("SVM: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_SVM))))

plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
plt.tight_layout()
plt.show()

"""Povećavanje hiperparametara gamma i C uzrokuje povećavanje točnosti na podacima za treniranje, 
no smanjenjem točnosti na podacima za testiranje. Potrebno je pronaći optimalne vrijednosti hiperparametara za najbolje rezultate.
Različiti tipovi kernela predstavljaju različite funkcije koje se koriste za granicu odluke. 
Tako različiti kerneli davaju različite točnosti uz iste vrijednosti hiperparametara."""


# Zadatak 6.5.4 
"""Pomocu unakrsne validacije odredite optimalnu vrijednost hiperparametra C i γ
 algoritma SVM za problem iz Zadatka 1."""

param_grid = {
    'C': [0.1, 1, 10, 100],       
    'gamma': [0.01, 0.1, 1, 10]  
}

grid_search = GridSearchCV(SVM_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train_n, y_train)

best_svm_model = grid_search.best_estimator_

best_C = grid_search.best_params_['C']
best_gamma = grid_search.best_params_['gamma']

test_accuracy = accuracy_score(y_test, best_svm_model.predict(X_test_n))

print(f"Najbolji model SVM (RBF kernel) postiže točnost od {test_accuracy:.3f} na skupu za testiranje.")
print(f"Optimalni hiperparametri: C={best_C}, gamma={best_gamma}")
