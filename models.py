import tensorflow as tf


from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import Input, layers



def lstm(xshape1, xshape2, optimizer):
    lstm = Sequential()
    lstm.add(LSTM(units=100, activation='tanh', name='first_lstm', input_shape=(xshape1, xshape2), return_sequences=True))
    lstm.add(LSTM(units=50, activation='tanh',return_sequences=True, dropout=0.2))
    lstm.add(LSTM(units=25, activation='tanh',return_sequences=True, dropout=0.2))
    lstm.add(LSTM(units=12, activation='tanh',dropout=0.2))
    lstm.add(Dense(1, activation="sigmoid"))

    lstm.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    return lstm


def conv1d(xshape1, xshape2, optimizer):
    model1 = Sequential()
    DROPOUT = 0.5
    model1.add(Conv1D(filters=64, kernel_size=2, activation='tanh',
               input_shape=(xshape1, xshape2)))
    model1.add(MaxPooling1D(pool_size=2))
    model1.add(Flatten())
    model1.add(Dense(100, activation='tanh'))
    model1.add(Dropout(DROPOUT))
    model1.add(Dense(50, activation='tanh'))
    model1.add(Dropout(DROPOUT))
    model1.add(Dense(25, activation='tanh'))
    model1.add(Dropout(DROPOUT))
    model1.add(Dense(1, activation='sigmoid'))
    model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model1


def conv1d2(xshape1,xshape2,optimizer):
    x = Sequential()
    x.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(xshape1,xshape2)))
    x.add(MaxPooling1D(pool_size=2))
    x.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'))
    x.add(MaxPooling1D(pool_size=2))
    x.add(Conv1D(filters=64, kernel_size=2, padding='same', activation='relu'))
    x.add(MaxPooling1D(pool_size=2))
    x.add(GRU(64))
    x.add(Dropout(0.5))
    x.add(Dense(10, activation="relu"))
    x.add(Dense(1, activation="sigmoid"))
    x.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return x



def super_complicated(xshape1,xshape2,optimizer):
    from tensorflow.keras.models import Model

    input_layer = Input(shape=(xshape1,xshape2))
    conv1 = Conv1D(filters=32,
                kernel_size=8,
                strides=1,
                activation='relu',
                padding='same')(input_layer)
    lstm1 = LSTM(32, return_sequences=True)(conv1)
    output_layer = Dense(1, activation='sigmoid')(lstm1)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def conv1dlstm(xshape1,xshape2,optimizer):
        model = Sequential()
        model.add(Conv1D(filters=256, kernel_size=5, padding='same', activation='relu',
                         input_shape=(xshape1,xshape2)))
        model.add(MaxPooling1D(pool_size=4))
        model.add(LSTM(64))

        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.50))
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.50))
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(0.50))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=["accuracy"])

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
    conv1d.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return conv1d


def lstm_autoencoder(xshape1, xshape2, optimizer):
    model = Sequential()

    model.add(LSTM(100, activation='tanh', input_shape=(xshape1, xshape2), return_sequences=True))
    model.add(LSTM(50, activation='tanh', return_sequences=False))
    model.add(RepeatVector(xshape1))
    model.add(LSTM(50, activation='tanh', return_sequences=True))
    model.add(LSTM(100, activation='tanh', return_sequences=True))
    model.add(TimeDistributed(Dense(1)))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    return model



def bilstm(xshape1, xshape2, optimizer):
    lstm = Sequential()
    lstm.add(Bidirectional(LSTM(units=100, activation='tanh', name='first_lstm',return_sequences=True, input_shape=(xshape1, xshape2),dropout=0.2))) #, return_sequences=True)))
    lstm.add(Bidirectional(LSTM(units=50, activation='tanh',return_sequences=True, dropout=0.2)))
    lstm.add(Bidirectional(LSTM(units=25, activation='tanh',return_sequences=True, dropout=0.2)))
    lstm.add(Bidirectional(LSTM(units=12, activation='tanh',dropout=0.2)))
    lstm.add(Dense(1, activation="sigmoid"))

    lstm.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    return lstm


def bigru(xshape1, xshape2, optimizer):
    lstm = Sequential()
    lstm.add(Bidirectional(GRU(units=100, activation='tanh', name='first_lstm', input_shape=(xshape1, xshape2), return_sequences=True)))
    lstm.add(Bidirectional(GRU(units=50, activation='tanh',return_sequences=True, dropout=0.2)))
    lstm.add(Bidirectional(GRU(units=25, activation='tanh',return_sequences=True, dropout=0.2)))
    lstm.add(Bidirectional(GRU(units=12, activation='tanh',dropout=0.2)))
    lstm.add(Dense(1, activation="sigmoid"))

    lstm.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])


    return lstm


def hypertune_bilstm(xshape1, xshape2):
    def build_model(hp):
        lstm = Sequential()
        droupout = hp.Choice('dropout',values=[0.2,0.3,0.4,0.5])
        lstm.add(Bidirectional(LSTM(units=hp.Int('num_of_neurons',min_value=50,max_value=100,step=10), activation='tanh', name='first_lstm', input_shape=(xshape1, xshape2), return_sequences=True)))
        lstm.add(Bidirectional(LSTM(units=50, activation='tanh',return_sequences=True, dropout=droupout)))
        lstm.add(Bidirectional(LSTM(units=25, activation='tanh',return_sequences=True, dropout=droupout)))
        lstm.add(Bidirectional(LSTM(units=12, activation='tanh',dropout=droupout)))
        lstm.add(Dense(1, activation="sigmoid"))
        lstm.compile(loss='binary_crossentropy', optimizer=SGD(hp.Choice('learning_rate', values =[1e-2,1e-3])), metrics=['accuracy'])
        return lstm
    return build_model


def hypertune_conv1d(xshape1,xshape2):
    def build_model(hp):
        dropout = hp.Choice('dropout',values=[0.5])
        lr = hp.Choice('lru',values=[0.1,0.01,0.001])
        optimizer = SGD(learning_rate=lr, momentum=0.9,
                    nesterov=True, clipnorm=1.)
        density = hp.Choice('dense',values=[100,200,300,400,500])
        model1 = Sequential()
        model1.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                input_shape=(xshape1, xshape2)))
        model1.add(MaxPooling1D(pool_size=2))
        model1.add(Flatten())
        model1.add(Dense(density, activation='relu'))
        model1.add(Dropout(dropout))
        model1.add(Dense(density//2, activation='relu'))
        model1.add(Dropout(dropout))
        model1.add(Dense(density//4, activation='relu'))
        model1.add(Dropout(dropout))
        model1.add(Dense(1, activation='sigmoid'))
        model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model1
    return build_model