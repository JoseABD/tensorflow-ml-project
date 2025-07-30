# Objetivo: Construir y entrenar una red neuronal multicapa para clasificar imágenes de dígitos manuscritos (MNIST)

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Cargar el dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizar imágenes a valores entre 0 y 1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Convertir etiquetas a one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Definición del modelo
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilación del modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluación del modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Precisión en datos de prueba: {test_acc:.4f}")

# Resultado esperado: Precisión entre 96% y 98%
