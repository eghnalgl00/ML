import tensorflow as tf
import numpy as np

X_train = np.array([[1,2],[3,4],[5,6],[7,8]])
X_mean = np.mean(X_train , axis = 0)
X_std = np.std(X_train)
X_norm =( X_train - X_mean) / X_std


y_train = np.array([1,0,0,1])

layer_1 = tf.keras.layers.Dense(units = 25, activation = "relu")
layer_2 = tf.keras.layers.Dense(units = 15, activation = "relu")
layer_3 = tf.keras.layers.Dense(units = 1, activation = "linear")

model = tf.keras.Sequential([layer_1,layer_2,layer_3])

model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits = True))
model.fit(X_norm, y_train , epochs = 100)


prob = tf.nn.sigmoid(model(np.array([[5,6]])))
predicted_class = np.argmax(prob[0])
print(y_train[predicted_class])

 

