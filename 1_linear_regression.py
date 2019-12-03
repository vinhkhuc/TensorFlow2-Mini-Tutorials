import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import Loss, MSE
from tensorflow.keras.optimizers import SGD


def build_model() -> Sequential:
    model = Sequential()
    model.add(Dense(1, input_shape=(1, ), use_bias=False, name="linear"))
    return model


def train(model: Sequential, loss: Loss, optimizer: SGD, x: tf.Tensor, y: tf.Tensor) -> float:

    with tf.GradientTape() as tape:
        # Forward
        x = tf.reshape(x, shape=(x.shape[0], 1))
        y_pred = model(x)
        y_pred = tf.squeeze(y_pred)
        output = tf.reduce_mean(loss(y, y_pred))

        # Backward
        grad = tape.gradient(output, model.trainable_variables)

        # Update parameters
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

    return output.numpy()


def main():
    tf.random.set_seed(42)
    X = tf.linspace(-1., 1., 101)
    Y = 2 * X + tf.random.uniform(X.shape) * 0.33

    model = build_model()
    model.summary()

    loss = MSE
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    batch_size = 10

    for i in range(50):
        cost = 0.
        num_batches = len(X) // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
            cost += train(model, loss, optimizer, X[start:end], Y[start:end])
        print("Epoch = %d, cost = %s" % (i + 1, cost / num_batches))

    w = model.variables[0]  # model has only one parameter
    print("w = %.2f" % w.numpy())  # will be approximately 2


if __name__ == "__main__":
    main()
