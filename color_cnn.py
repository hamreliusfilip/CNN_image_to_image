from tensorflow.keras import layers, models
from tensorflow import keras
import util
import data_generator

def conv_block(x, N, channels, kernel_size, activation, padding='same', kernel_reg=None):
    for i in range(N):
        x = layers.Conv2D(channels, 
                          kernel_size=kernel_size, 
                          activation=activation, 
                          padding=padding,
                          kernel_regularizer=kernel_reg)(x)   
        x = layers.BatchNormalization()(x)
    return layers.MaxPooling2D(pool_size=(2, 2))(x)

def colorization_cnn(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # First convolution block (Grayscale -> Feature extraction)
    x = conv_block(inputs, N=2, channels=32, kernel_size=(3, 3), activation='relu', padding='same')
    
    # Second convolution block
    x = conv_block(x, N=2, channels=64, kernel_size=(3, 3), activation='relu', padding='same')
    
    # Third convolution block
    x = conv_block(x, N=2, channels=128, kernel_size=(3, 3), activation='relu', padding='same')
    
    # Fourth convolution block for more detailed features
    x = conv_block(x, N=2, channels=256, kernel_size=(3, 3), activation='relu', padding='same')
    
    # Final layer to output an RGB image
    output = layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')(x)
    
    model = models.Model(inputs, output)
    return model

# Assuming grayscale input shape
input_shape = (height, width, 1)  # Grayscale image
model = colorization_cnn(input_shape)

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()
