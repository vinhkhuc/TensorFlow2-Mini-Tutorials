import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.losses import Loss, CategoricalCrossentropy
from tensorflow.keras.optimizers import SGD

from data_util import load_mnist


class LSTMNet(Model):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = LSTM(hidden_dim, return_state=True)
        self.linear = Dense(output_dim, input_shape=(hidden_dim, ), use_bias=False)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        batch_size = x.shape[0]
        h0 = tf.zeros(shape=(batch_size, self.hidden_dim))
        c0 = tf.zeros(shape=(batch_size, self.hidden_dim))

        fx, hn, cn = self.lstm(x, initial_state=(h0, c0))
        return self.linear(fx)


def train(model: Model, loss: Loss, optimizer: SGD, x: tf.Tensor, y: tf.Tensor) -> float:

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

    train_size = len(trY)
    n_classes = 10
    seq_length = 28
    input_dim = 28
    hidden_dim = 128
    batch_size = 100
    epochs = 20

    # Convert to the shape (num_samples, seq_length, input_dim)
    trX = trX.reshape(-1, seq_length, input_dim)
    teX = teX.reshape(-1, seq_length, input_dim)

    trX = tf.convert_to_tensor(trX, dtype=tf.float32)
    teX = tf.convert_to_tensor(teX, dtype=tf.float32)
    trY = tf.convert_to_tensor(trY, dtype=tf.float32)

    model = LSTMNet(input_dim, hidden_dim, n_classes)
    # Pass input_shape for building the model so that model.summary() works
    # The input_shape's 1st component is the batch size. The dummy value 1 is used here.
    model.build(input_shape=(1, seq_length, input_dim))
    model.summary()

    loss = CategoricalCrossentropy(from_logits=True)
    optimizer = SGD(learning_rate=0.01, momentum=0.9)

    for i in range(epochs):
        cost = 0.
        num_batches = train_size // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer, trX[start:end, :, :], trY[start:end])
        predY = predict(model, teX)
        print("Epoch %d, cost = %f, acc = %.2f%%" %
              (i + 1, cost / num_batches, 100. * np.mean(predY == teY.argmax(axis=1))))


if __name__ == "__main__":
    main()
