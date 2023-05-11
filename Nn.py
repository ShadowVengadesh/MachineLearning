import tensorflow as tf 

from tensorflow import keras 
# Load the MNIST dataset 

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() 

# Normalize the input data 

x_train = x_train / 255.0 

x_test = x_test / 255.0 

# Define the model architecture 

model = keras.Sequential([ 

 keras.layers.Flatten(input_shape=(28, 28)), 

 keras.layers.Dense(128, activation='relu'), 

 keras.layers.Dense(10, activation='softmax') 

] ) 

# Compile the model 

model.compile(optimizer='adam', 

 loss='sparse_categorical_crossentropy', 

 metrics=['accuracy']) 

# Train the model 

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test)) 

# Evaluate the model 

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2) 

print('Test accuracy:', test_acc)
