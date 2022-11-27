import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers as layers
from tensorflow.keras.layers import GlobalMaxPooling2D, Activation, Dense, Conv1D, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras import regularizers

def get_model(input_length):
    inspected_chanels= 8
    input_layer = keras.Input(shape = (inspected_chanels,input_length), name='input')
 
    x = tf.transpose(input_layer, perm=[0, 2, 1])

    x = layers.Reshape((input_length,inspected_chanels,1))(x)


    l2 =0.001
    conv_params = 20 

    x = layers.Conv2D(conv_params, kernel_size=(50,4), padding='same', activation='elu', kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.MaxPooling2D(pool_size=(10,1))(x)
    x = layers.Dropout(.1)(x)



    memory = 500
    x = layers.Reshape((memory, int((x.shape[1] *x.shape[2]*x.shape[3])/memory)))(x)
    x  = tf.keras.layers.LSTM(20, kernel_regularizer=regularizers.l2(l2), dropout = 0.1, recurrent_dropout = 0.)(x)
    x = layers.Dropout(.1)(x)


    
    #x= layers.Dense(10, kernel_regularizer=regularizers.l2(l2))(x)



    
    output = layers.Dense(3, activation = 'softmax', name = 'output')(x)

       

    model = keras.Model(inputs=input_layer, outputs=output)

    model.summary()
    return model