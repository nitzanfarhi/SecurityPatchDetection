import tensorflow as tf


from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import Input, layers


def small_conv1d(xshape1, xshape2):
    conv1d = Sequential()
    conv1d.add(Conv1D(filters=64, kernel_size=3, activation='relu',
               input_shape=(xshape1, xshape2)))
    conv1d.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    conv1d.add(Dropout(0.5))
    conv1d.add(MaxPooling1D(pool_size=2))
    conv1d.add(Flatten())
    conv1d.add(Dense(100, activation='relu'))
    conv1d.add(Dense(1, activation='sigmoid'))
    conv1d.compile(loss='binary_crossentropy',
                   optimizer='adam', metrics=['accuracy'])

    return conv1d


def lstm(xshape1, xshape2, optimizer):
    lstm = Sequential()
    lstm.add(LSTM(units=50, activation='relu', name='first_lstm', recurrent_dropout=0.1,
                  input_shape=(xshape1, xshape2)))
    lstm.add(Dense(1, activation="sigmoid"))

    lstm.compile(loss='binary_crossentropy',
                 optimizer=optimizer,  metrics=['accuracy'])

    return lstm


def conv1d(xshape1, xshape2):
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
    model1.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   metrics=['accuracy'])

    return model1


def conv1d_more_layers(xshape1, xshape2):
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
    conv1d.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                   metrics=['accuracy'])

    return conv1d


def lstm_autoencoder(xshape1, xshape2):
    model = Sequential()

    lstm_autoencoder.add(LSTM(32, activation='relu', input_shape=(
        xshape1, xshape2), return_sequences=True))
    lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=False))
    lstm_autoencoder.add(RepeatVector(xshape1))
    # Decoder
    lstm_autoencoder.add(LSTM(16, activation='relu', return_sequences=True))
    lstm_autoencoder.add(LSTM(32, activation='relu', return_sequences=True))
    lstm_autoencoder.add(TimeDistributed(Dense(1)))

    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                  metrics=['accuracy'])

    return model
