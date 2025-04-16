import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

X = generate_data(500, 1)

plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()

km = KMeans(n_clusters=2, init='random', n_init=5, random_state=0)

km.fit(X)

labels = km.predict(X)

plt.figure()
plt.scatter(X[:,0],X[:,1], c=labels, cmap='viridis')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('K = 2')
plt.colorbar(label='Grupa')
plt.show()

km = KMeans(n_clusters=4, init='random', n_init=5, random_state=0)
km.fit(X)
labels = km.predict(X)

plt.figure()
plt.scatter(X[:,0],X[:,1], c=labels, cmap='viridis')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('K = 4')
plt.colorbar(label='Grupa')
plt.show()

km = KMeans(n_clusters=5, init='random', n_init=5, random_state=0)
km.fit(X)
labels = km.predict(X)

plt.figure()
plt.scatter(X[:,0],X[:,1], c=labels, cmap='viridis')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('K = 5')
plt.colorbar(label='Grupa')
plt.show()


km = KMeans(n_clusters=3, init='random', n_init=5, random_state=0)
km.fit(X)
labels = km.predict(X)

plt.figure()
plt.scatter(X[:,0],X[:,1], c=labels, cmap='viridis')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Način 1 - 3 grupe')
plt.colorbar(label='Grupa')
plt.show()

# generiranje podatkovnih primjera (način 2)
X = generate_data(500, 2)
km = KMeans(n_clusters=3, init='random', n_init=5, random_state=0)
km.fit(X)
labels = km.predict(X)

plt.figure()
plt.scatter(X[:,0],X[:,1], c=labels, cmap='viridis')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Način 2 - 3 grupe')
plt.colorbar(label='Grupa')
plt.show()

# generiranje podatkovnih primjera (način 3)
X = generate_data(500, 3)
km = KMeans(n_clusters=4, init='random', n_init=5, random_state=0)
km.fit(X)
labels = km.predict(X)

plt.figure()
plt.scatter(X[:,0],X[:,1], c=labels, cmap='viridis')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Način 3 - 4 grupe')
plt.colorbar(label='Grupa')
plt.show()

# generiranje podatkovnih primjera (način 4)
X = generate_data(500, 4)
km = KMeans(n_clusters=2, init='random', n_init=5, random_state=0)
km.fit(X)
labels = km.predict(X)

plt.figure()
plt.scatter(X[:,0],X[:,1], c=labels, cmap='viridis')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Način 4 - 2 grupe')
plt.colorbar(label='Grupa')
plt.show()

# generiranje podatkovnih primjera (način 5)
X = generate_data(500, 5)
km = KMeans(n_clusters=2, init='random', n_init=5, random_state=0)
km.fit(X)
labels = km.predict(X)

plt.figure()
plt.scatter(X[:,0],X[:,1], c=labels, cmap='viridis')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Način 5 - 2 grupe')
plt.colorbar(label='Grupa')
plt.show()

"""Algoritam K srednjih vrijednosti daje dobre rezultate samo u slučaju 
kada su podaci sferično raspoređeni i kompaktni. Također algoritam 
pretpostavlja da su varijance svake grupe slične te da su grupe linearno razdvojive."""