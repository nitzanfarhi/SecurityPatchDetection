import tensorflow as tf
from tensorflow import keras

from keras.layers import BatchNormalization

from keras.layers import Dense, LSTM, Input, Flatten, MaxPool1D
from keras.layers import Dense, LSTM, GRU, BatchNormalization
from keras.layers import  Convolution1D
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras import Sequential
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.layers import Bidirectional
from keras import Input
from keras.models import Model

def lstm(xshape1, xshape2, optimizer):
    lstm = Sequential()
    lstm.add(LSTM(units=100, activation='tanh',
             input_shape=(xshape1, xshape2), return_sequences=True))
    lstm.add(LSTM(units=50, activation='tanh',
             return_sequences=True, dropout=0.2))
    lstm.add(LSTM(units=25, activation='tanh',
             return_sequences=True, dropout=0.2))
    lstm.add(LSTM(units=12, activation='tanh', dropout=0.2))
    lstm.add(Dense(1, activation="sigmoid"))

    lstm.compile(loss='binary_crossentropy',
                 optimizer=optimizer, metrics=['accuracy'])

    return lstm


def gru(xshape1, xshape2, optimizer):
    lstm = Sequential()
    lstm.add(GRU(units=100, activation='tanh', name='first_lstm',
             input_shape=(xshape1, xshape2), return_sequences=True))
    lstm.add(GRU(units=50, activation='tanh',
             return_sequences=True, dropout=0.2))
    lstm.add(GRU(units=25, activation='tanh',
             return_sequences=True, dropout=0.2))
    lstm.add(GRU(units=12, activation='tanh', dropout=0.2))
    lstm.add(Dense(1, activation="sigmoid"))

    lstm.compile(loss='binary_crossentropy',
                 optimizer=optimizer, metrics=['accuracy'])

    return lstm

def conv1d2(xshape1, xshape2, optimizer):
    x = Sequential()
    x.add(Conv1D(filters=64, kernel_size=2, padding='same',
          activation='relu', input_shape=(xshape1, xshape2)))
    x.add(MaxPooling1D(pool_size=2))
    x.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'))
    x.add(MaxPooling1D(pool_size=2))
    x.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'))
    x.add(MaxPooling1D(pool_size=2))
    x.add(GRU(64))
    x.add(Dropout(0.5))
    x.add(Dense(10, activation="relu"))
    x.add(Dense(1, activation="sigmoid"))
    x.compile(loss='binary_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])
    return x


def super_complicated(xshape1, xshape2, optimizer):

    input_layer = Input(shape=(xshape1, xshape2))
    conv1 = Conv1D(filters=32,
                   kernel_size=8,
                   strides=1,
                   activation='relu',
                   padding='same')(input_layer)
    lstm1 = LSTM(32, return_sequences=True)(conv1)
    output_layer = Dense(1, activation='sigmoid')(lstm1)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def conv1d_more_layers(xshape1, xshape2, optimizer):
    conv1d = Sequential()
    conv1d.add(Conv1D(filters=500, kernel_size=2, activation='relu',
               input_shape=(xshape1, xshape2)))
    conv1d.add(Dropout(0.1))
    conv1d.add(MaxPooling1D(pool_size=2))
    conv1d.add(Dropout(0.1))
    conv1d.add(Flatten())
    conv1d.add(Dense(100, activation='relu'))
    conv1d.add(Dropout(0.1))
    conv1d.add(Dense(100, activation='relu'))
    conv1d.add(Dropout(0.1))
    conv1d.add(Dense(100, activation='relu'))
    conv1d.add(Dropout(0.1))
    conv1d.add(Dense(70, activation='relu'))
    conv1d.add(Dropout(0.1))
    conv1d.add(Dense(50, activation='relu'))
    conv1d.add(Dropout(0.1))
    conv1d.add(Dense(30, activation='relu'))
    conv1d.add(Dropout(0.1))
    conv1d.add(Dense(1, activation='sigmoid'))
    conv1d.compile(loss='binary_crossentropy',
                   optimizer=optimizer, metrics=['accuracy'])

    return conv1d


def lstm_autoencoder(xshape1, xshape2, optimizer):
    model = Sequential()

    model.add(LSTM(100, activation='tanh', input_shape=(
        xshape1, xshape2), return_sequences=True))
    model.add(LSTM(50, activation='tanh', return_sequences=False))
    model.add(RepeatVector(xshape1))
    model.add(LSTM(50, activation='tanh', return_sequences=True))
    model.add(LSTM(100, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    return model


def bilstm(xshape1, xshape2, optimizer):
    lstm = Sequential()
    lstm.add(Bidirectional(LSTM(units=100, activation='tanh', name='first_lstm', return_sequences=True,
             input_shape=(xshape1, xshape2), dropout=0.2)))  # , return_sequences=True)))
    lstm.add(Bidirectional(LSTM(units=50, activation='tanh',
             return_sequences=True, dropout=0.2)))
    lstm.add(Bidirectional(LSTM(units=25, activation='tanh',
             return_sequences=True, dropout=0.2)))
    lstm.add(Bidirectional(LSTM(units=12, activation='tanh', dropout=0.2)))
    lstm.add(Dense(1, activation="sigmoid"))

    lstm.compile(loss='binary_crossentropy',
                 optimizer=optimizer, metrics=['accuracy'])

    return lstm


def bigru(xshape1, xshape2, optimizer):
    lstm = Sequential()
    lstm.add(Bidirectional(GRU(units=100, activation='tanh', name='first_lstm',
             input_shape=(xshape1, xshape2), return_sequences=True)))
    lstm.add(Bidirectional(GRU(units=50, activation='tanh',
             return_sequences=True, dropout=0.2)))
    lstm.add(Bidirectional(GRU(units=25, activation='tanh',
             return_sequences=True, dropout=0.2)))
    lstm.add(Bidirectional(GRU(units=12, activation='tanh', dropout=0.2)))
    lstm.add(Dense(1, activation="sigmoid"))

    lstm.compile(loss='binary_crossentropy',
                 optimizer=optimizer, metrics=['accuracy'])

    return lstm


# def hypertune_bilstm(xshape1, xshape2):
#     def build_model(hp):
#         lstm = Sequential()
#         droupout = hp.Choice('dropout', values=[0.2, 0.3, 0.4, 0.5])
#         lstm.add(Bidirectional(LSTM(units=hp.Int('num_of_neurons', min_value=50, max_value=100, step=10),
#                  activation='tanh', name='first_lstm', input_shape=(xshape1, xshape2), return_sequences=True)))
#         lstm.add(Bidirectional(LSTM(units=50, activation='tanh',
#                  return_sequences=True, dropout=droupout)))
#         lstm.add(Bidirectional(LSTM(units=25, activation='tanh',
#                  return_sequences=True, dropout=droupout)))
#         lstm.add(Bidirectional(
#             LSTM(units=12, activation='tanh', dropout=droupout)))
#         lstm.add(Dense(1, activation="sigmoid"))
#         lstm.compile(loss='binary_crossentropy', optimizer=SGD(
#             hp.Choice('learning_rate', values=[1e-2, 1e-3])), metrics=['accuracy'])
#         return lstm
#     return build_model


def gru_cnn_bad_1(xshape1, xshape2, optimizer='adam', recurrent_units=100, dropout_rate=0.2, recurrent_dropout_rate=0.2, dense_size=100):
        # Conv1d LSTM: Accuracy - 

    input_layer = Input(shape=(xshape1,xshape2), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Dropout(dropout_rate)(input_layer) 
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=recurrent_units, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = GRU(recurrent_units)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_size, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 


def gru_cnn_bad_2(xshape1, xshape2, optimizer='adam', recurrent_units=100, dropout_rate=0.2, recurrent_dropout_rate=0.2, dense_size=100):
        # Conv1d LSTM: Accuracy - 
    inp = Input(shape=(xshape1,xshape2), name='mfcc_in')
    model = inp

    model = Conv1D(filters=12, kernel_size=(3), activation='relu')(model)
    model = Conv1D(filters=12, kernel_size=(3), activation='relu')(model)
    model = Flatten()(model)

    model = Dense(56)(model)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)
    model = Dropout(0.2)(model)
    model = Dense(28)(model)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)

    model = Dense(1)(model)
    model = Activation('sigmoid')(model)

    model = Model(inp, model)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model 

def gru_cnn_bad_3(xshape1, xshape2, optimizer='adam', recurrent_units=100, dropout_rate=0.2, recurrent_dropout_rate=0.2, dense_size=100):
    
    inputs = Input(shape=(xshape1, xshape2), name='input_layer')   
    x=inputs
    for dilation_rate in [4,3,2,1]:
        x = Conv1D(filters=12,
               kernel_size=3, 
               padding='causal',
               dilation_rate=dilation_rate,
               activation='linear')(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    #x = Dense(7, activation='relu', name='dense_layer')(x)
    outputs = Dense(1, activation='sigmoid', name='output_layer')(x)
    model =  Model(inputs, outputs=[outputs])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
def gru_cnn_bad_4(xshape1, xshape2, optimizer='adam', recurrent_units=100, dropout_rate=0.2, recurrent_dropout_rate=0.2, dense_size=100):

    model = Sequential()

    model.add(Conv1D(filters=256, kernel_size=3, strides=1, activation='relu', input_shape=(99, 40), name='block1_conv1'))
    model.add(MaxPool1D(pool_size=2, name='block1_pool1'))
    model.add(BatchNormalization(momentum=0.9, epsilon=1e-5, axis=1))

    model.add(Conv1D(filters=256, kernel_size=3, strides=1, activation='relu', name='block1_conv2'))
    model.add(MaxPool1D(pool_size=2, name='block1_pool2'))

    model.add(Flatten(name='block1_flat1'))
    model.add(Dropout(0.5, name='block1_drop1'))

    model.add(Dense(512, activation='relu', name='block2_dense2'))
    # model.add(MaxoutDense(512, nb_feature=4, name="block2_maxout2"))
    model.add(Dropout(0.5, name='block2_drop2'))

    model.add(Dense(512, activation='relu', name='block2_dense3'))
    # model.add(MaxoutDense(512, nb_feature=4, name="block2_maxout3"))
    model.add(Dense(1, activation='sigmoid', name="predict"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def gru_cnn_bad_5(xshape1, xshape2, optimizer='adam', recurrent_units=100, dropout_rate=0.2, recurrent_dropout_rate=0.2, dense_size=100):
    
    inp = Input(shape=(xshape1, xshape2))
    model = inp
    
    model = Conv1D(filters=24, kernel_size=(3), activation='relu')(model)
    model = LSTM(16)(model)
    model = Activation('relu')(model)
    model = Dropout(0.2)(model)
    model = Dense(16)(model)
    model = Activation('relu')(model)
    model = BatchNormalization()(model)
    model = Dense(1)(model)
    model = Activation('sigmoid')(model)
    
    model = Model(inp, model)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def gru_cnn_bad_6(xshape1, xshape2, optimizer='adam', recurrent_units=100, dropout_rate=0.2, recurrent_dropout_rate=0.2, dense_size=100):
    #inp = Input(shape=(maxlen, ))
    input_layer = Input(shape=(xshape1, xshape2), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Dropout(dropout_rate)(input_layer) 
    x = Conv1D(filters=256, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(input_layer) 
    x = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(dropout_rate)(input_layer) 
    x = Conv1D(filters=64, kernel_size=2, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = GRU(recurrent_units, return_sequences=False, dropout=dropout_rate, activation='tanh')(x)
    #x = Dropout(dropout_rate)(x)
    x = Dense(dense_size, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 

def gru_cnn_bad_7(xshape1, xshape2, optimizer='adam', recurrent_units=100, dropout_rate=0.2, recurrent_dropout_rate=0.2, dense_size=100):
    filter_kernels = [7, 7, 5, 5, 3, 3]
    #inp = Input(shape=(maxlen, ))
    input_layer = Input(shape=(xshape1, xshape2), )
    #x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    conv = Conv1D(nb_filter=recurrent_units, filter_length=filter_kernels[0], activation='relu')(input_layer)
    conv = MaxPooling1D(pool_length=3)(conv)
    conv1 = Conv1D(nb_filter=recurrent_units, filter_length=filter_kernels[1], activation='relu')(conv)
    conv1 = MaxPooling1D(pool_length=3)(conv1)
    conv2 = Conv1D(nb_filter=recurrent_units, filter_length=filter_kernels[2], activation='relu')(conv1)
    conv3 = Conv1D(nb_filter=recurrent_units, filter_length=filter_kernels[3], activation='relu')(conv2)
    conv4 = Conv1D(nb_filter=recurrent_units, filter_length=filter_kernels[4], activation='relu')(conv3)
    conv5 = Conv1D(nb_filter=recurrent_units, filter_length=filter_kernels[5], activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_length=3)(conv5)
    conv5 = Flatten()(conv5)
    z = Dropout(0.5)(Dense(dense_size, activation='relu')(conv5))
    #x = GlobalMaxPool1D()(x)
    x = Dense(1, activation="sigmoid")(z)
    model = Model(inputs=input_layer, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def gru_cnn_bad_8(xshape1, xshape2, optimizer='adam', recurrent_units=100, dropout_rate=0.2, recurrent_dropout_rate=0.2, dense_size=100):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(xshape1, xshape2),
        dropout=dropout_rate, activation='tanh',
        return_sequences=True
    ))
    model.add(LSTM(512, dropout=dropout_rate, activation='tanh', return_sequences=True))
    model.add(LSTM(512, dropout=dropout_rate, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])
    return model

def gru_cnn_bad_9(xshape1, xshape2, optimizer='adam', recurrent_units=100, dropout_rate=0.2, recurrent_dropout_rate=0.2, dense_size=100):
    regressor = Sequential()
    # First LSTM layer with Dropout regularisation
    regressor.add(LSTM(units=32, return_sequences=True, input_shape=(xshape1, xshape2)))
    regressor.add(Dropout(0.2))
    # Second LSTM layer
    regressor.add(LSTM(units=64, return_sequences=True))
    regressor.add(Dropout(0.2))
    # Third LSTM layer
    regressor.add(LSTM(units=128, return_sequences=True))
    regressor.add(Dropout(0.5))
    # Fourth LSTM layer
    regressor.add(LSTM(units=256))
    regressor.add(Dropout(0.5))
    # The output layer
    regressor.add(Dense(units=1, activation='sigmoid'))
    regressor.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])
    return regressor

def lstm_cnn(xshape1, xshape2, optimizer='adam', recurrent_units=100, dropout_rate=0.2, recurrent_dropout_rate=0.2, dense_size=100):
    model = Sequential()
    model.add(Convolution1D(64, 3, input_shape= (xshape1,xshape2)))
    model.add(MaxPooling1D())
    model.add(LSTM(100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Convolution1D(32, 3))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])
    return model

def gru_cnn(xshape1, xshape2, optimizer='adam', recurrent_units=100, dropout_rate=0.2, recurrent_dropout_rate=0.2, dense_size=100):
    model = Sequential()
    model.add(Convolution1D(256, 3, input_shape= (xshape1,xshape2)))
    model.add(MaxPooling1D())
    model.add(GRU(300, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Convolution1D(128, 3))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics = ['accuracy'])
    return model

def hypertune_gru(xshape1, xshape2):
    def build_model(hp):
        recurrent_units = hp.Choice('recurrent_units', values=[100, 200, 300])
        dropout_rate = hp.Choice('dropout_rate', values=[0.2, 0.3, 0.4])
        recurrent_dropout_rate = hp.Choice(
            'recurrent_dropout_rate', values=[0.2, 0.3, 0.4])
        dense_size = hp.Choice('dense_size', values=[100, 200, 300])
        return gru_cnn(xshape1, xshape2, recurrent_units=recurrent_units, dropout_rate=dropout_rate, recurrent_dropout_rate=recurrent_dropout_rate, dense_size=dense_size)

    return build_model


def conv1d(xshape1, xshape2, optimizer):
    model1 = Sequential()
    DROPOUT = 0.3
    model1.add(Conv1D(filters=256, kernel_size=2, activation='tanh',
               input_shape=(xshape1, xshape2)))

    model1.add(MaxPooling1D(pool_size=2))
    model1.add(Flatten())
    model1.add(Dense(1024, activation='tanh'))
    model1.add(Dropout(DROPOUT))
    model1.add(Dense(256, activation='tanh'))
    model1.add(Dropout(DROPOUT))
    model1.add(Dense(64, activation='tanh'))
    model1.add(Dropout(DROPOUT))
    model1.add(Dense(1, activation='sigmoid'))
    model1.compile(loss='binary_crossentropy',
                   jit_compile=True, steps_per_execution=100,
                   optimizer=Adagrad(
                       learning_rate=0.01), metrics=['accuracy'])

    return model1


def hypertune_conv1d(xshape1, xshape2):
    # "steps_per_execution": 150,
    # "dropout": 0.2,
    # "filters1": 128,
    # "dense1": 256,
    # "dense2": 256,
    # "dense3": 64,
    # "learning_rate": 0.01,
    # "optimizer": "adagrad",
    # "tuner/epochs": 100,
    # "tuner/initial_epoch": 0,
    # "tuner/bracket": 0,
    # "tuner/round": 0
    def build_model(hp):
        spe = hp.Choice('steps_per_execution', values=[100, 150, 200])
        DROPOUT = hp.Choice('dropout', values=[0.2, 0.3, 0.4, 0.5])
        filters1 = hp.Choice('filters1', values=[64, 128, 256])
        dense1 = hp.Choice('dense1', values=[256, 512, 1024])
        dense2 = hp.Choice('dense2', values=[64, 128, 256])
        dense3 = hp.Choice('dense3', values=[64, 128, 256])
        lr = hp.Choice('learning_rate', values=[1e-2, 1e-3])

        # Select optimizer    
        optimizer=hp.Choice('optimizer', values=['adam', 'adagrad', 'SGD'])

        # Conditional for each optimizer
        if optimizer == 'adam':
            opt = Adam(learning_rate=lr)    
        elif optimizer == 'adagrad':
            opt = Adagrad(learning_rate=lr)
        elif optimizer == 'SGD':
            opt = SGD(learning_rate=lr)
        elif optimizer == 'RMSprop':
            opt = RMSprop(learning_rate=lr)
        
        model1 = Sequential()
        model1.add(Conv1D(filters=filters1, kernel_size=2, activation='tanh',
                input_shape=(xshape1, xshape2)))

        model1.add(MaxPooling1D(pool_size=2))
        model1.add(Flatten())
        model1.add(Dense(dense1, activation='tanh'))
        model1.add(Dropout(DROPOUT))
        model1.add(Dense(dense2, activation='tanh'))
        model1.add(Dropout(DROPOUT))
        model1.add(Dense(dense3, activation='tanh'))
        model1.add(Dropout(DROPOUT))
        model1.add(Dense(1, activation='sigmoid'))
        model1.compile(loss='binary_crossentropy',
                    jit_compile=True, steps_per_execution=spe,optimizer=opt, metrics=['accuracy'])


        return model1
    return build_model


def hypertune_gru_cnn(xshape1, xshape2):
    def build_model(hp):
        spe = hp.Choice('steps_per_execution', values=[100, 150, 200])
        DROPOUT = hp.Choice('dropout', values=[0.2, 0.3, 0.4, 0.5])
        dense1 = hp.Choice('dense1', values=[256, 512, 1024])
        dense2 = hp.Choice('dense2', values=[64, 128, 256])
        gru1 = hp.Choice('gru1', values=[64, 128, 256])
        lr = hp.Choice('learning_rate', values=[1e-2, 1e-3])

        # Select optimizer    
        optimizer=hp.Choice('optimizer', values=['adam', 'adagrad', 'SGD'])

        # Conditional for each optimizer
        if optimizer == 'adam':
            opt = Adam(learning_rate=lr)    
        elif optimizer == 'adagrad':
            opt = Adagrad(learning_rate=lr)
        elif optimizer == 'SGD':
            opt = SGD(learning_rate=lr)
        elif optimizer == 'RMSprop':
            opt = RMSprop(learning_rate=lr)
        
        model = Sequential()
        model.add(Convolution1D(dense1, 3, input_shape= (xshape1,xshape2)))
        model.add(MaxPooling1D())
        model.add(GRU(gru1, return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(Convolution1D(dense2, 3))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                    jit_compile=True, steps_per_execution=spe,optimizer=opt, metrics=['accuracy'])


        return model
    return build_model

def hypertune_gru_cnn(xshape1, xshape2):
    def build_model(hp):
        spe = hp.Choice('steps_per_execution', values=[100, 150, 200])
        DROPOUT = hp.Choice('dropout', values=[0.2, 0.3, 0.4, 0.5])
        dense1 = hp.Choice('dense1', values=[256, 512, 1024])
        dense2 = hp.Choice('dense2', values=[64, 128, 256])
        gru1 = hp.Choice('gru1', values=[64, 128, 256])
        lr = hp.Choice('learning_rate', values=[1e-2, 1e-3])

        # Select optimizer    
        optimizer=hp.Choice('optimizer', values=['adam', 'adagrad', 'SGD'])

        # Conditional for each optimizer
        if optimizer == 'adam':
            opt = Adam(learning_rate=lr)    
        elif optimizer == 'adagrad':
            opt = Adagrad(learning_rate=lr)
        elif optimizer == 'SGD':
            opt = SGD(learning_rate=lr)
        elif optimizer == 'RMSprop':
            opt = RMSprop(learning_rate=lr)
        
        model = Sequential()
        model.add(Convolution1D(dense1, 3, input_shape= (xshape1,xshape2)))
        model.add(MaxPooling1D())
        model.add(GRU(gru1, return_sequences=True))
        model.add(Dropout(DROPOUT))
        model.add(Convolution1D(dense2, 3))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                    jit_compile=True, steps_per_execution=spe,optimizer=opt, metrics=['accuracy'])


        return model
    return build_model


  
  
def hypertune_gru(xshape1, xshape2):
    def build_model(hp):
        DROPOUT = hp.Choice('dropout', values=[0.2, 0.3, 0.4, 0.5])
        gru1 = hp.Choice('gru1', values=[64, 128, 256])
        gru2 = hp.Choice('gru1', values=[64, 128, 256])
        gru3 = hp.Choice('gru1', values=[64, 128, 256])
        lr = hp.Choice('learning_rate', values=[1e-2, 1e-3])

        # Select optimizer    
        optimizer=hp.Choice('optimizer', values=['adam', 'adagrad', 'SGD'])

        # Conditional for each optimizer
        if optimizer == 'adam':
            opt = Adam(learning_rate=lr)    
        elif optimizer == 'adagrad':
            opt = Adagrad(learning_rate=lr)
        elif optimizer == 'SGD':
            opt = SGD(learning_rate=lr)
        elif optimizer == 'RMSprop':
            opt = RMSprop(learning_rate=lr)
        
        lstm = Sequential()
        lstm.add(Bidirectional(GRU(units=gru1, activation='tanh', name='first_lstm',
                input_shape=(xshape1, xshape2), return_sequences=True)))
        lstm.add(Bidirectional(GRU(units=gru2, activation='tanh',
                return_sequences=True, dropout=DROPOUT)))
        lstm.add(Bidirectional(GRU(units=gru3, activation='tanh',
                return_sequences=False, dropout=DROPOUT)))
        lstm.add(Dense(1, activation="sigmoid"))

        lstm.compile(loss='binary_crossentropy',
                    optimizer=optimizer, metrics=['accuracy'])

        return lstm
    return build_model
