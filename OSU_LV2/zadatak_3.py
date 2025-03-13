import numpy as np
import matplotlib.pyplot as plt

image = plt.imread("road.jpg")
image = image[:,:,0].copy()

height, width = image.shape

second_quarter_image = image[:, width // 4 : width // 2]
rotated_image = np.rot90(image, k=3)
flipped_image = np.flipud(image)

plt.figure(figsize=(10, 6))

plt.subplot(231), plt.imshow(image, cmap="gray"), plt.title('Originalna slika')
plt.subplot(232), plt.imshow(image, alpha=0.6, cmap="gray"), plt.title('Posvijetljena slika')
plt.subplot(233), plt.imshow(second_quarter_image, cmap="gray"), plt.title('Druga četvrtina slike')
plt.subplot(234), plt.imshow(rotated_image, cmap="gray"), plt.title('Rotirana slika')
plt.subplot(235), plt.imshow(flipped_image, cmap="gray"), plt.title('Zrcaljena slika')

plt.show()