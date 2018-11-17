import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.datasets import fashion_mnist


def load_data(data=fashion_mnist.load_data(), num_classes=10):
    """Load data and return training and testing data.

    data        : The data to load
    num_classes : The number of classes for the classifier

    returns     : The training and testing data
    """

    (x_train, y_train), (x_test, y_test) = data

    # Reshape
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # Normalisation
    x_train = x_train / 255
    x_test = x_test / 255

    # One hot encode
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def build_model(data_shape, num_classes=10):
    """Build and return the CNN model.

    Architecture:
        Conv2D->pool->Conv2D->pool->dropout->flatten->dense128->dense50->dense10->softmax

    returns : The model
    """

    model = Sequential()
    model.add(
        Conv2D(30, (5, 5), input_shape=data_shape[1:], activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    model.summary()

    return model


def data_augment(x_train):
    """Augment data with rotation=15, hori flip, width/height shift=0.1.

    We augment data to train a better network.
    """

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)

    datagen.fit(x_train)

    return datagen


def train_model(x_train, y_train, x_test, y_test, model, datagen):
    """Train model, batch size=64, epochs=50."""

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=100,
        batch_size=64)

    model.save('fashion_mnist.h5')

    return model


def test_model(x_test, y_test, model):
    """Testing the model."""

    scores = model.evaluate(x_test, y_test, batch_size=128, verbose=1)
    print('\nTest result: %.3f loss: %.3f' % (scores[1] * 100, scores[0]))


def main():

    # Load data
    x_train, y_train, x_test, y_test = load_data()

    # Build model
    data_shape = x_train.shape
    model = build_model(data_shape)

    # Augment data and train model
    datagen = data_augment(x_train)
    model = train_model(x_train, y_train, x_test, y_test, model, datagen)

    # Evaluate model
    test_model(x_test, y_test, model)


if __name__ == "__main__":
    main()
