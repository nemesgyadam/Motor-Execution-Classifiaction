import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers as layers
from tensorflow.keras.layers import GlobalMaxPooling2D, Activation, Dense, Conv1D, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras import regularizers

def get_model(input_length):
    inspected_chanels= 8
    input_layer = keras.Input(shape = (inspected_chanels,input_length, 1), name='input')


    
    l2 =0.0001
  
    x = layers.Conv2D(8, kernel_size=(1,32), padding='same', activation='elu', kernel_regularizer=regularizers.l2(l2))(input_layer)
    x = layers.BatchNormalization()(x)

    #x = layers.Conv2D(32, kernel_size=(1,16), padding='same', activation='elu', kernel_regularizer=regularizers.l2(l2))(x)
    
    #x = layers.Dropout(.2)(x)

    x = layers.Conv2D(8, kernel_size=(8,1), padding='same', activation='elu', kernel_regularizer=regularizers.l2(l2))(x)
    #x = layers.Conv2D(16, kernel_size=(4,1), padding='same', activation='elu', kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)


    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dropout(.1)(x)
    #x = layers.Flatten()(x)
    

    #x = layers.Dense(32, activation='elu', kernel_regularizer=regularizers.l2(l2))(x)
    #x = layers.BatchNormalization()(x)

    output = layers.Dense(3,activation = 'softmax', name = 'out')(x)
    x = layers.Dropout(.1)(x)
    model = keras.Model(inputs=input_layer, outputs=output)

    model.summary()
    return model