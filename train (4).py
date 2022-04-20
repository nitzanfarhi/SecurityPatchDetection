#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install tqdm')
get_ipython().system('pip install -U scikit-learn scipy matplotlib')


# In[6]:


import os

import datetime as dt
import pandas as pd
import numpy as np
import pandas as pd
import tqdm
import random
import pylab
import random

from datetime import datetime, timedelta
from dateutil import parser
from numpy import array
from tensorflow.keras.layers import Dense, LSTM,GRU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Sequential
from collections import Counter
from pandas import DataFrame
from enum import Enum
from matplotlib import pyplot

get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        seq_x = np.pad(seq_x, ((0, 0), (0, 30 - seq_x.shape[1])), 'constant')
        X.append(seq_x)
        y.append(seq_y[-1])
    return array(X), array(y)


# In[9]:


from matplotlib import pyplot as plt
def draw_timeline(name,vulns,first_date, last_date):

    dates = vulns
    dates += [first_date]
    dates += [last_date]

    values = [1]*len(dates)
    values[-1] = 2
    values[-2] = 2

    X = pd.to_datetime(dates)
    fig, ax = plt.subplots(figsize=(6,1))
    ax.scatter(X, [1]*len(X), c=values,
               marker='s', s=100)
    fig.autofmt_xdate()

    # everything after this is turning off stuff that's plotted by default
    ax.set_title(name)
    ax.yaxis.set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_facecolor('white')
    
    ax.get_yaxis().set_ticklabels([])
    # day = pd.to_timedelta("1", unit='D')
    # plt.xlim(X[0] - day, X[-1] + day)
    plt.show()
    #plt.subplots_adjust(bottom=0.15)
    #plt.savefig(f"D:/cve/images/timeline/{name}.jpg", transparent=False)
    
    
def find_benign_events(cur_repo_data,gap_days, num_of_events):
    benign_events = []
    retries = num_of_events * 5
    counter = 0
    for _ in range(num_of_events):
        found_event = False
        while not found_event:
            if counter >=retries:
                return benign_events
            try:
                cur_event = random.randint(2*gap_days+1,cur_repo_data.shape[0]-gap_days*2-1)
            except ValueError:
                counter +=1
                continue
            event = cur_repo_data.index[cur_event]

            before_vuln = event - gap_days
            after_vuln = event + gap_days
            res_event = cur_repo_data[before_vuln:event-1]
            if not res_event[res_event["VulnEvent"]>0].empty:
                counter +=1
                continue
            benign_events.append(res_event.iloc[:,:-1].values)
            found_event = True
            
            
    return benign_events

def create_all_events(cur_repo_data,gap_days):
    all_events = []
    labels = []
    for i in range(gap_days,cur_repo_data.shape[0],1):
            event = cur_repo_data.index[i]
            before_vuln = event - gap_days
            res_event = cur_repo_data[before_vuln:event-1]
            all_events.append(res_event.iloc[:,:-1].values)
            labels.append(res_event.iloc[:,-1].values)
    return all_events,labels


def add_time_one_hot_encoding(df):
    # print(df.index.day_of_week)
    # print(df.index.hour)
    hour = pd.get_dummies(df.index.hour.astype(pd.CategoricalDtype(categories=range(24))),prefix='hour')
    week = pd.get_dummies(df.index.day_of_week.astype(pd.CategoricalDtype(categories=range(7))),prefix='day_of_week')
    day_of_month = pd.get_dummies(df.index.day.astype(pd.CategoricalDtype(categories=range(1,32))),prefix='day_of_month')

    df = pd.concat([df.reset_index(),hour,week,day_of_month],axis=1)
    df = df.set_index('index')
    return df

def get_event_window(cur_repo_data, event, aggr_options, days=10, hours=10,resample=24):
    starting_time = event - timedelta(days=days,hours=hours)

    if aggr_options == Aggregate.before_cve:
        res = cur_repo_data[starting_time:event]
        res = res.iloc[:-1,:]
        res = res.resample(f'{resample}H').sum()

    elif aggr_options == Aggregate.after_cve:
        res = cur_repo_data[starting_time:event]
        res = res.iloc[:-1,:]
        new_row = pd.DataFrame([[0]*len(res.columns)], columns = res.columns, index=[starting_time])
        res = pd.concat([new_row,res], ignore_index=False)
        res = res.resample(f'{resample}H').sum()
    else:
        res = cur_repo_data.reset_index(drop=True)
        res = res[event-num_of_events-1:event-1]

    return add_time_one_hot_encoding(res)

def find_best_f1(X_test,y_test,model):
    max_f1 = 0
    thresh = 0
    best_y = 0
    pred = model.predict(X_test)
    for i in range(100):
        y_predict = (pred.reshape(-1)>i/1000).astype(int)
        precision, recall, fscore, support = score(y_test,y_predict ,zero_division=0)
        cur_f1 = fscore[1]
        # print(i,cur_f1)
        if cur_f1 > max_f1:
            max_f1 = cur_f1
            best_y = y_predict
            thresh = i / 100
    return max_f1,thresh, best_y


# In[10]:


repo_dirs = '../repo_gharchive_processed4'
benign_all, vuln_all = [], []
n_features = 0
gap_days = 150

nice_list= ['facebook_hhvm.csv',
'ffmpeg_ffmpeg.csv',
'flatpak_flatpak.csv',
'freerdp_freerdp.csv',
'git_git.csv',
'gpac_gpac.csv',
'imagemagick_imagemagick.csv',
'kde_kdeconnect-kde.csv',
'krb5_krb5.csv',
'mantisbt_mantisbt.csv',
'op-tee_optee_os.csv',
'owncloud_core.csv',
'php_php-src.csv',
'revive-adserver_revive-adserver.csv',
'rubygems_rubygems.csv',
'the-tcpdump-group_tcpdump.csv']

class Aggregate(Enum):
    none = 1
    before_cve = 2
    after_cve = 3
    
aggr_options = Aggregate.after_cve

num_of_events = 10
days = 50
hours = 0
resample = 24
benign_vuln_ratio = 10


for file in os.listdir(repo_dirs)[:]:
    try:
        selected = 4
        if file not in nice_list[:]:
                  continue
        cur_repo_data = pd.read_csv(repo_dirs + "/" + file,parse_dates=['created_at'],index_col='created_at')

        if cur_repo_data.shape[0]<100:
            continue

        cur_repo_data = cur_repo_data[cur_repo_data.index.notnull()]
        cur_repo_data["additions"]=(cur_repo_data["additions"]-cur_repo_data["additions"].mean())/cur_repo_data["additions"].std()
        cur_repo_data["deletions"]=(cur_repo_data["deletions"]-cur_repo_data["deletions"].mean())/cur_repo_data["deletions"].std()

    except pd.errors.EmptyDataError:
        continue


    cols_at_end = ['VulnEvent']
    cur_repo_data = cur_repo_data[[c for c in cur_repo_data if c not in cols_at_end]
                            + [c for c in cols_at_end if c in cur_repo_data]]

    vulns = cur_repo_data.index[cur_repo_data['VulnEvent'] > 0].tolist()
    benigns = cur_repo_data.index[cur_repo_data['VulnEvent'] == 0].tolist()
    random.shuffle(benigns)
    for vuln in vulns:
        res = get_event_window(cur_repo_data,vuln,aggr_options,days=days,hours=hours,resample=resample)
        vuln_all.append(res.values)
    print(file)
    benign_counter = 0
    for benign in tqdm.tqdm(benigns):
        if benign_counter >= benign_vuln_ratio*len(vulns):
            break

        res = get_event_window(cur_repo_data,benign,aggr_options,days=days,hours=hours,resample=resample)
        benign_all.append(res.values)            
        benign_counter+=1
    print(file)


# In[8]:


max_vals = max(Counter([v.shape for v in vuln_all]))
vuln_all = [v for v in vuln_all if v.shape == max_vals]
max_vals = max(Counter([v.shape for v in benign_all]))
benign_all = [v for v in benign_all if v.shape == max_vals]

vuln_all =np.nan_to_num(np.array(vuln_all))
benign_all = np.nan_to_num(np.array(benign_all)) 
name_template = f"{str(aggr_options)}_{benign_vuln_ratio}_H{hours}_D{days}_R{resample}"
vuln_npy_name = name_template+"_vuln.npy"
benign_npy_name = name_template+"_benign.npy"

np.save("ready_data/"+vuln_npy_name, np.array(vuln_all))    # .npy extension is added if not given
np.save("ready_data/"+benign_npy_name, np.array(benign_all))    # .npy extension is added if not given


# In[9]:



vuln_all = np.load("ready_data/"+vuln_npy_name)
benign_all = np.load("ready_data/"+benign_npy_name)
def normalize(time_series_feature):
    if time_series_feature.max()-time_series_feature.min() == 0:
        return time_series_feature
    return (time_series_feature-time_series_feature.min())/(time_series_feature.max()-time_series_feature.min())


# In[10]:


all_train_x = np.concatenate([vuln_all,benign_all])
all_train_y = np.concatenate([np.ones(vuln_all.shape[0]),np.zeros(benign_all.shape[0])])
all_train_x.shape,all_train_y.shape

NORMALIZE = True

if NORMALIZE:
    all_train_x= normalize(all_train_x)
    vuln_all = normalize(vuln_all)
    benign_all = normalize(benign_all)


# In[ ]:


from sklearn.model_selection import train_test_split
import tensorflow as tf
print(all_train_x.shape, all_train_x[0].shape)
print(all_train_y.shape, all_train_y[0].shape)
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras import Input, layers
from tensorflow.keras.callbacks import EarlyStopping

X_train, X_test, y_train, y_test = train_test_split(all_train_x,all_train_y,shuffle=True)

model = Sequential()
#model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Flatten())
#model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

reshaped_train,reshaped_test = X_train.reshape(X_train.shape[0],-1), X_test.reshape(X_test.shape[0],-1)

# define model
model = Sequential()
model.add(Conv1D(filters=100, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.8))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(70, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

es = EarlyStopping(monitor='val_accuracy', mode='max',patience=30)

import itertools


def generator(feat,labels):
    pairs = [(x, y) for x in feat for y in labels]
    cycle_pairs = itertools.cycle(pairs)
    while (True):
        f, p = next(cycle_pairs)
        return np.array([f]), np.array([p])

history = model.fit_generator(generator(X_train, y_train), verbose=1,epochs=50, batch_size=16,validation_data=(X_test,y_test),callbacks=[])

# print(model.evaluate(X_test.reshape(X_test.shape[0],-1), y_test, verbose=0))
pyplot.plot(history.history['accuracy'])
pyplot.plot(history.history['val_accuracy'])
pyplot.title('model accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.legend(['train', 'val'], loc='upper left')
pyplot.show()


from sklearn.metrics import precision_recall_fscore_support as score



f1,thresh,best_y = find_best_f1(X_test,y_test,model)
print(f1)

