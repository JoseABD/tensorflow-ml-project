As an introductory example, a simple linear regression 
model can be built to learn the relationship between two 
variables. Suppose the goal is to model the function Y = 2X 
using a small dataset. With TensorFlow, a network is 
defined with a single layer and one neuron, which, after 
several training iterations, successfully learns the linear 
relationship. When testing the model with a new input, 
such as X = 10, the expected output is approximately 20, 
thereby validating that it has correctly captured the 
underlying pattern. Although elementary, this type of 
exercise is essential for understanding how TensorFlow 
represents and fits models, and it lays the groundwork for
developing more sophisticated architectures in real-world 
contexts.

step by step:

STEP 1:

import tensorflow as tf

import numpy as np
- tensorflow: a powerful library for building and training machine learning models.
- numpy: helps manage numerical arrays and mathematical operations efficiently


STEP 2:

X = np.array([1.0,2.0, 3.0, 4.0], dtype=float)

Y = np.array([2.0, 4.0, 6.0, 8.0], dtype=float)
- X: input values.
- Y: target output values.

These follow the linear relationship Y = 2X.


STEP 3:

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
- Sequential: creates a stack of layers.
- Dense: a fully connected layer with:
  - units=1: one neuron.
  - input_shape=[1]: expects one input value. This layer learns a weight and bias to model the linear function.


STEP 4:

model.compile(optimizer='sgd', loss='mean_squared_error')
- optimizer='sgd': uses stochastic gradient descent to adjust weights.
- loss='mean_squared_error': calculates how far predictions are from actual results.


STEP 5:

model.fit(X, Y, epochs=500, verbose=0)
- Trains the model for 500 epochs.
- verbose=0: suppresses training output (quiet mode).


STEP 6:

result = model.predict(np.array([10.0]))

print(f"Prediction for X=10: {result[0][0]:.2f}")
- Tests the model with a new value X = 10.0.
- The expected output is around 20.0 because the model learned Y = 2X.
