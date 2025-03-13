import numpy as np
import matplotlib.pyplot as plt

file = open('data.csv')

array = np.array(np.loadtxt(file, delimiter=',', skiprows=1))

# a)
print(f"Mjerenje je izvršeno na {array.shape[0]} osoba.")
# b)
plt.scatter(x=array[:,1], y=array[:,2], marker='o', c='red')
plt.xlabel("Height")
plt.ylabel("Mass")
plt.show()
# c)
plt.scatter(x=array[49::50,1], y=array[49::50,2], marker='o', c='red')
plt.xlabel("Height")
plt.ylabel("Mass")
plt.show()
# d)
print(array[:,1].astype(float).min())
print(array[:,1].astype(float).max())
print(array[:,1].astype(float).mean())
# e)
ind = array[:,0].astype(float) == 1.0
print("Men: ")
print(f"Min: {array[ind,1].astype(float).min()}")
print(f"Max: {array[ind,1].astype(float).max()}")
print(f"Mean: {array[ind,1].astype(float).mean()}")

ind = array[:,0].astype(float) == 0.0
print("Women: ")
print(f"Min: {array[ind,1].astype(float).min()}")
print(f"Max: {array[ind,1].astype(float).max()}")
print(f"Mean: {array[ind,1].astype(float).mean()}")