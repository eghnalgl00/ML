import tensorflow as tf
import numpy as np
import pandas as pd

df = pd.read_csv("animal_classification.csv")
X_train = df[['weight_kg', 'height_cm', 'lifespan_years', 'top_speed_kmh']]
y_train = df["label"]


mean = np.mean(X_train, axis=0) 
std  = np.std(X_train, axis=0)  

X_norm = (X_train - mean) / std

layer_1 = tf.keras.layers.Dense(25, activation="relu")
layer_2 = tf.keras.layers.Dense(15, activation="relu")
layer_3 = tf.keras.layers.Dense(3, activation="linear")

model = tf.keras.Sequential([layer_1, layer_2, layer_3])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

model.fit(X_norm, y_train, epochs=100)

logit = model(X_norm)
f_x = tf.nn.softmax(logit)

animal_example = pd.DataFrame({"weight_kg" : [8] , 'height_cm' : [35], 'lifespan_years' : [14] , 'top_speed_kmh' : [42] })
animal_example_norm = (animal_example - mean) / std 

prob = tf.nn.softmax(model(animal_example_norm))
predicted_class = np.argmax(prob[0])

print(f"Predicted class: {predicted_class}")          
print(f"Confidence: {prob[0][predicted_class]:.4f}")
print(f"All probabilities: {prob}") 

