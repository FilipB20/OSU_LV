import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
for i in range(1,7):
    img = Image.imread(f"imgs/test_{i}.jpg")

    # prikazi originalnu sliku
    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255
    # transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    unique_colors = np.unique(img_array, axis=0)
    num_unique_colors = len(unique_colors)
    print(f"Broj različitih boja u originalnoj slici: {num_unique_colors}")

    # rezultatna slika
    img_array_aprox = img_array.copy()

    # Primjena K-means algoritma za grupiranje boja
    kmeans = KMeans(n_clusters=5, init='random', n_init=5, random_state=42)  
    kmeans.fit(img_array_aprox)

    # Dobivanje centara grupa (boja)
    cluster_centers = kmeans.cluster_centers_

    labels = kmeans.labels_ #indeksi grupe svakog piksela

    for i in range(len(img_array)):
        img_array_aprox[i] = cluster_centers[labels[i]]

    # Oblikovanje slike natrag u originalne dimenzije
    img_quantized = np.reshape(img_array_aprox, (w, h, d))

    # Prikaz rezultantne slike
    plt.figure()
    plt.title("Kvantizirana slika s 5 boja")
    plt.imshow(img_quantized)
    plt.tight_layout()
    plt.show()

    k_values = range(1, 11)  

    inertia_values = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(img_array)
        inertia_values.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertia_values, marker='o', linestyle='-', color='b')
    plt.xlabel('Broj grupa (K)')
    plt.ylabel('Inercija')
    plt.title('Ovisnost inercije o broju grupa (K)')
    plt.grid(True)
    plt.show()

    K = 5
    for i in range(K):
        binary_img = np.zeros((w*h, d))  
        binary_img[labels == i] = 1  
        binary_img = np.reshape(binary_img, (w, h, d))
        plt.figure()
        plt.imshow(binary_img, cmap='gray')  
        plt.title(f'Grupa {i+1}')
        plt.axis('off')
        plt.show()

"""Na primjeru prve slike dovoljno je čak uzeti dvije boje za dobivanje 
rezultantne slike jer se radi o tekstu kojeg je potrebno moći pročitati. 
Što se više boja uzima to je slika realnija i čitljivija, no nakon nekog 
broja boja (npr. 5) razlike su zanemarive."""

"""Svaka od boja prikazana je bijelom bojom dok su sve ostale prikazane crnom. 
Vidi se gdje na rezultantnoj slici leži koja boja."""