import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt


# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]),plt.yticks([])
    plt.imshow(X_train[i])

plt.show()


# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# CNN mreza
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", 
                                  patience = 5, 
                                  verbose = 1),
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn_dropout',
                                update_freq = 100)
]

learning_rate = 1.0

optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

n_samples = 25000  

X_train_small = X_train_n[:n_samples]
y_train_small = y_train[:n_samples]

model.fit(X_train_small,
            y_train_small,
            epochs = 40,
            batch_size = 64,
            callbacks = my_callbacks,
            validation_split = 0.1)


score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')

# Zadatak 9.4.1
""" 1) Mreža se sastoji od tri konvolucijska sloja, tri sloja sažimanja i 
dva potpuno povezana sloja od 500 i 10 neurona. Mreža ima 1,122,758 parametara.
2) Tijekom procesa treniranja, nakon određenog broja epoha, preciznost na validacijskom skupu počinje padati, 
dok funkcija gubitka raste. Na testnom skupu mreža postiže točnost od 73.46%."""

# Zadatak 9.4.2
"""Uvođenjem dropout sloja poboljšane su vrijednosti točnosti klasifikacije i funkcije gubitka 
na validacijskom skupu. Dropout pomaže jer smanjuje preveliku ovisnost mreže o pojedinačnim značajkama, 
čime se smanjuje rizik od prenaučenosti. Na testnom skupu postignuta točnost iznosi 74.89%. """


#  Zadatak 9.4.3
""" Došlo je do ranog zaustavljanja u 11. epohi. Točnost na testnom skupu je 75.73."""

# Zadatak 9.4.4
""" 1) Jako velika veličina serije: Može ubrzati proces učenja jer će se gradijenti 
izračunati na većem uzorku podataka, ali istovremeno može dovesti do veće 
potrošnje memorije i računalnih resursa.
Jako mala veličina serije: Može dovesti do bržeg konvergiranja jer se 
češće ažuriraju težine mreže, ali može biti osjetljiva na šum u podacima 
i može zahtijevati dulje vrijeme učenja.
2) Jako mala stopa učenja: Može dovesti do sporog konvergiranja ili zapinjanja u 
lokalnim minimumima jer su koraci ažuriranja težina mali.
Jako velika stopa učenja: Može uzrokovati divergenciju ili osciliranje u 
učenju jer su koraci ažuriranja preveliki i mogu "preskočiti" minimum.
3) Izbačeni slojevi mogu dovesti do pojednostavljivanja modela i smanjenja 
potrebnih resursa za treniranje. Međutim, izbacivanje slojeva može dovesti 
do gubitka složenosti i sposobnosti modela da nauči složenije obrasce u podacima.
4) Smanjenje veličine skupa za učenje može dovesti do bržeg učenja jer će se manji 
skup podataka brže obrađivati. No, smanjenje veličine skupa za učenje također može 
dovesti do gubitka sposobnosti modela da generalizira, posebno ako se uklone važni 
dijelovi podataka ili se ne pokriju svi aspekti problema.

Zaljučak: Kombiniranje i pravilno podešavanje ovih faktora ključno je za 
uspješno treniranje neuronskih mreža."""