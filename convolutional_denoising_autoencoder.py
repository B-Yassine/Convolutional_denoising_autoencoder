from keras.layers import Input, Conv2D, Reshape, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt


(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

input_img = Input(shape=(28, 28, 1))

#Encoding
x = Conv2D(16, (5, 5), activation='relu', padding='same')(input_img)
x = Conv2D(8, (5, 5), activation='relu', padding='same')(x)

#Decoding
x = Conv2DTranspose(16, (5, 5), activation='relu', padding='same')(x)
x = Conv2DTranspose(8, (5, 5), activation='relu', padding='same')(x)


decoded = Conv2D(1, (5, 5), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adam', metrics=['accuracy'], loss="binary_crossentropy")

autoencoder.fit(x_train_noisy, x_train,
                epochs=5,
                batch_size=64,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

autoencoder.save("denoising_autoencoder.h5")
#autoencoder.load_weights("denoising_autoencoder.h5")

decoded_imgs = autoencoder.predict(x_test_noisy, verbose=1)

n = 15
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i +1+ n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

score = autoencoder.evaluate(x_test, decoded_imgs, verbose=1)
print('Test Loss:', score[0])
