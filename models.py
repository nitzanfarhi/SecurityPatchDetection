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
    model1.add(Conv1D(filters=64, kernel_size=2, activation='relu',
               input_shape=(xshape1, xshape2)))
    model1.add(MaxPooling1D(pool_size=2))
    model1.add(Flatten())
    model1.add(Dense(100, activation='relu'))
    model1.add(Dropout(0.50))
    model1.add(Dense(50, activation='relu'))
    model1.add(Dropout(0.50))
    model1.add(Dense(25, activation='relu'))
    model1.add(Dropout(0.50))
    model1.add(Dense(1, activation='sigmoid'))
    model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model1


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
    lstm.add(Bidirectional(LSTM(units=100, activation='tanh', name='first_lstm', input_shape=(xshape1, xshape2),dropout=0.5))) #, return_sequences=True)))
    lstm.add(Bidirectional(LSTM(units=50, activation='tanh',return_sequences=True, dropout=0.5)))
    lstm.add(Bidirectional(LSTM(units=25, activation='tanh',return_sequences=True, dropout=0.5)))
    lstm.add(Bidirectional(LSTM(units=12, activation='tanh',dropout=0.5)))
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

