import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

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
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']
input_variables2 = ['bill_length_mm',
                   'bill_depth_mm',
                   'body_mass_g',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

#a)
unique_train, counts_train = np.unique(y_train, return_counts=True)
class_counts_train = dict(zip(unique_train, counts_train))

unique_test, counts_test = np.unique(y_test, return_counts=True)
class_counts_test = dict(zip(unique_test, counts_test))

plt.figure(figsize=(10, 6))
plt.bar(class_counts_train.keys(), class_counts_train.values(), color='blue', alpha=0.5, label='Train')
plt.bar(class_counts_test.keys(), class_counts_test.values(), color='red', alpha=0.5, label='Test')
plt.xlabel('Klasa')
plt.ylabel('Broj primjera')
plt.title('Broj primjera po klasi u skupu za učenje i testiranje')
plt.xticks(list(labels.keys()), labels.values())
plt.legend()
plt.grid(True)
plt.show()

#b)
model = LogisticRegression()
model.fit(X_train, y_train)

#c)
coefficients = model.coef_
print("Koeficijenti modela:")
print(coefficients)

intercepts = model.intercept_
print("Parametri odsječka na y-osi:")
print(intercepts)

for i, label in labels.items():
    coefficients_class = coefficients[i]
    print(f"Koeficijenti za klasu {label}: {coefficients_class}")

for i, label in labels.items():
    intercept_class = intercepts[i]
    print(f"Koeficijenti za klasu {label}: {intercept_class}")

#d)
#plot_decision_regions(X_train, y_train.ravel(), model)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Granice odluke modela logističke regresije na skupu za učenje')
plt.legend()
plt.show()

#e)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

y_pred = model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Matrica zabune:")
print(conf_matrix)

accuracy = accuracy_score(y_test, y_pred)
print("Točnost: {:.2f}".format(accuracy))

report = classification_report(y_test, y_pred, target_names=list(labels.values()))
print("Classification Report:")
print(report)


#f)
"""Classification Report (za input_variables2):
              precision    recall  f1-score   support

      Adelie       1.00      0.93      0.96        27
   Chinstrap       0.89      1.00      0.94        17
      Gentoo       1.00      1.00      1.00        25

    accuracy                           0.97        69
   macro avg       0.96      0.98      0.97        69
weighted avg       0.97      0.97      0.97        69

Classification Report (za input_variables):
              precision    recall  f1-score   support

      Adelie       0.96      0.89      0.92        27
   Chinstrap       0.94      0.88      0.91        17
      Gentoo       0.89      1.00      0.94        25

    accuracy                           0.93        69
   macro avg       0.93      0.92      0.93        69
weighted avg       0.93      0.93      0.93        69

Klasifikacija je malo bolja s 4 ulazne veličine nego s 2, no nije značajna razlika."""