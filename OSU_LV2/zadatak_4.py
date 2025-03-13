import numpy as np
import matplotlib.pyplot as plt

white = np.zeros((50, 50))
black = np.ones((50,50))

black_white = np.hstack((black, white))
white_black = np.hstack((white, black))

result = np.vstack((white_black,black_white))

plt.figure(figsize=(10, 6))
plt.imshow(result, cmap="gray")
plt.show()