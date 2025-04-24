from PIL import Image
import numpy as np
import keras
from matplotlib import pyplot as plt
from keras import models

model = models.load_model('mnist_model.keras')

for i in range(0, 10):
    image_path = f'test{i}.png'
    image = Image.open(image_path)

    plt.imshow(image)
    plt.title('Originalna slika')
    plt.show()

    image = image.convert('L')  
    image = image.resize((28, 28))  

    plt.imshow(image)
    plt.show()

    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255
    image_array = np.expand_dims(image_array, axis=0)  

    prediction = model.predict(image_array)
    predicted_label = np.argmax(prediction)

    print(f"Predviđena oznaka slike '{image_path}': {predicted_label}")

"""Model pogodi rijetko koju znamenku, vjerojatno jer je naučen na specifičan rukopis."""