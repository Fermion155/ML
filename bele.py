import tensorflow as tf
import tensorflow.keras.backend as K
import os
import numpy as np

y_true = tf.constant([[[[3.0, 1.0], [2.0, 2.0]], [[2.0, 2.0], [6.0, 6.0]]], [[[3.0, 3.0], [7.0, 7.0]], [[4.0, 4.0], [5.0, 6.0]]]])
print(y_true[0][1])

#intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
#sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
suma = tf.keras.backend.sum(y_true, axis=[0,-1, -2])
print(suma)
print(tf.keras.backend.mean(suma))

metric = 0.0
pred = 1.2
true = 0

if true == 0:
 metric += pred
 print("Cia")

print(pred)
print(metric)

arr = np.arange(21)
print(arr)
np.random.shuffle(arr)
print(arr)