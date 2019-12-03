import tensorflow as tf

a = tf.constant([2, 3, 4])
b = tf.constant([3, 4, 5])
m = a * b
print(m.numpy())  # outputs [ 6 12 20]
