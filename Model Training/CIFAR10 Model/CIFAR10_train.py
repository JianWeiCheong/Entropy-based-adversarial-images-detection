import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras import regularizers, optimizers
from keras.datasets import cifar10


def load_data(data=cifar10.load_data(), num_classes=10):
    """Load data and return training and testing data.

    data        : The data to load
    num_classes : The number of classes for the classifier

    returns     : The training and testing data
    """

    (x_train, y_train), (x_test, y_test) = data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))

    # Normalisation
    # std + 1e-7 in case of divide by 0
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    # One hot encode
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def build_model(data_shape, num_classes=10, baseMapNum=32, weight_decay=1e-4):
    """Build and return the CNN model.

    Architecture:
        Conv2D->relu->BatchNormalise->Conv2D->relu->BatchNormalise->pool->dropout
        (x3)
        Flatten -> softmax

    dropouts    : 0.2 -> 0.3 -> 0.4
    mapnum      : base -> 2*base -> 4*base
    pooling     : 32 -> 16 -> 8 -> 4

    returns     : The model
    """

    model = Sequential()

    # Layers 1
    model.add(
        Conv2D(
            baseMapNum, (3, 3),
            padding='same',
            kernel_regularizer=regularizers.l2(weight_decay),
            input_shape=data_shape[1:]))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            baseMapNum, (3, 3),
            padding='same',
            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Layers 2
    model.add(
        Conv2D(
            2 * baseMapNum, (3, 3),
            padding='same',
            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            2 * baseMapNum, (3, 3),
            padding='same',
            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    # Layers 3
    model.add(
        Conv2D(
            4 * baseMapNum, (3, 3),
            padding='same',
            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            4 * baseMapNum, (3, 3),
            padding='same',
            kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Layers out
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))

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
    """Train model, batch size=64, epochs=200.

    Optimizer : rmsprop
    0-75 epochs    -> lr=0.001
    75-100 epochs  -> lr=0.0005
    100-125 epochs -> lr=0.0003
    """

    batch_size = 64
    epochs = 25
    lr = 0.001

    for i in range(0, 6):
        if i == 0:
            epoch = 3 * epochs
        else:
            epoch = epochs

        opt_rms = optimizers.rmsprop(lr=lr, decay=1e-6)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt_rms,
            metrics=['accuracy'])
        model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=x_train.shape[0] // batch_size,
            epochs=epoch,
            verbose=1,
            validation_data=(x_test, y_test))
        model.save('cifar10_opt_rms.h5')

        lr /= 2

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
