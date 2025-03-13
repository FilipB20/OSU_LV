import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,3,1])
y=np.array([1,2,2,1,1])

plt.plot(x,y,'r',linewidth =1 , marker ="*", markersize = 7)
plt.axis ([0 ,4 ,0 , 4])
plt.xlabel("X vrijednost")
plt.ylabel("Y vrijednost")
plt.title("Slika 2.3")
plt.show()
