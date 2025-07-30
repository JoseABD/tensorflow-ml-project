# Objetivo: Entrenar un modelo simple de regresión lineal que aprenda la función Y = 2X usando TensorFlow

import tensorflow as tf
import numpy as np

# Datos de entrada y salida
X = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
Y = np.array([2.0, 4.0, 6.0, 8.0], dtype=float)

# Definición del modelo: una sola capa densa con una neurona
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compilación del modelo
model.compile(optimizer='sgd', loss='mean_squared_error')

# Entrenamiento del modelo
model.fit(X, Y, epochs=500, verbose=0)

# Prueba del modelo con un nuevo valor
result = model.predict(np.array([10.0]))
print(f"Predicción para X=10: {result[0][0]:.2f}")

# Resultado esperado: Aproximadamente 20.0
