from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
from tensorflow.keras.losses import Loss, CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

from data_util import load_mnist


def build_model(input_shape: Tuple[int, int, int], output_dim: int) -> Sequential:
    # Sequential model
    model = Sequential()

    # With the input_shape= (N, 28, 28, 1), conv_1's shape = (N, 24, 24, 10)
    # Since MaxPool(Relu(x)) = Relu(MaxPool(x)), it doesn't matter what order we apply ReLU and MaxPool
    model.add(Conv2D(filters=10, kernel_size=(5, 5), input_shape=input_shape, activation="relu", name="conv_1"))

    # maxpool_1's shape = (N, 12, 12, 10)
    model.add(MaxPool2D(pool_size=(2, 2), name="maxpool_1"))

    # conv_2's shape = (N, 8, 8, 20)
    model.add(Conv2D(filters=20, kernel_size=(5, 5), input_shape=(12, 12, 10), activation="relu", name="conv_2"))

    # dropout_2's shape = (N, 4, 4, 20)
    model.add(Dropout(0.5, name="dropout_2"))

    # maxpool_2's shape = (N, 4, 4, 20)
    model.add(MaxPool2D(pool_size=(2, 2), name="maxpool_2"))

    # shape = (N, 4 * 4 * 20) = (N, 320)
    model.add(Flatten(name="flatten"))

    # fc_1's shape = (N, 50)
    model.add(Dense(50, input_shape=(320, ), activation="relu", name="fc_1"))
    model.add(Dropout(0.5, name="dropout_3"))

    # fc_2's shape = (N, output_dim)
    model.add(Dense(output_dim, input_shape=(50, ), name="fc_2"))

    return model


def train(model: Sequential, loss: Loss, optimizer: SGD, x: tf.Tensor, y: tf.Tensor) -> float:

    # Forward
    with tf.GradientTape() as tape:
        y_pred = model(x)
        output = tf.reduce_mean(loss(y, y_pred))

    # Backward
    grad = tape.gradient(output, model.trainable_variables)

    # Update parameters
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    return output.numpy()


def predict(model: Sequential, x: tf.Tensor) -> np.ndarray:
    output = model(x)
    return output.numpy().argmax(axis=1)


def main():
    tf.random.set_seed(42)
    trX, teX, trY, teY = load_mnist(onehot=True)
    trX = trX.reshape(-1, 28, 28, 1)
    teX = teX.reshape(-1, 28, 28, 1)
    n_examples = trX.shape[0]
    input_shape = trX.shape[1:]

    trX = tf.convert_to_tensor(trX, dtype=tf.float32)
    teX = tf.convert_to_tensor(teX, dtype=tf.float32)
    trY = tf.convert_to_tensor(trY, dtype=tf.float32)

    n_classes = 10
    model = build_model(input_shape, n_classes)
    model.summary()

    loss = CategoricalCrossentropy(from_logits=True)
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    batch_size = 100

    for i in range(20):
        cost = 0.
        num_batches = n_examples // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer, trX[start:end], trY[start:end])

        predY = predict(model, teX)
        print("Epoch %d, cost = %f, acc = %.2f%%"
              % (i + 1, cost / num_batches, 100. * np.mean(predY == teY.argmax(axis=1))))


if __name__ == "__main__":
    main()
