import numpy as np

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Loss, CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

from data_util import load_mnist


def build_model(input_dim: int, output_dim: int) -> Sequential:
    # We don't need the softmax layer here since
    # CategoricalCrossentropy has an option to accept logits directly.
    model = Sequential()
    model.add(Dense(output_dim, input_shape=(input_dim, ), use_bias=False, name="linear"))
    return model


# TODO: Support tf.function
# Current code throws exception with Tensor not having .numpy()
#@tf.function
def train(model: Sequential, loss: Loss, optimizer: SGD, x: tf.Tensor, y: tf.Tensor) -> float:

    # Forward
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
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
    trX = tf.convert_to_tensor(trX, dtype=tf.float32)
    teX = tf.convert_to_tensor(teX, dtype=tf.float32)
    trY = tf.convert_to_tensor(trY, dtype=tf.float32)

    n_examples, n_features = trX.shape
    n_classes = 10
    model = build_model(n_features, n_classes)
    model.summary()

    loss = CategoricalCrossentropy(from_logits=True)
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    batch_size = 100

    for i in range(50):
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
