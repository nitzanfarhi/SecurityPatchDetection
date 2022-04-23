#!/usr/bin/env python
# coding: utf-8

import os
import logging

import datetime as dt
import pandas as pd
import numpy as np
import pandas as pd
import tqdm
import random
import random
import itertools
from datetime import datetime, timedelta
from dateutil import parser
from numpy import array
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Sequential
from collections import Counter
from pandas import DataFrame
from enum import Enum
from matplotlib import pyplot
from matplotlib import pyplot as plt

from helper import normalize, find_best_f1, EnumAction

from sklearn.model_selection import train_test_split
import tensorflow as tf
import argparse
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


def find_benign_events(cur_repo_data, gap_days, num_of_events):
    benign_events = []
    retries = num_of_events * 5
    counter = 0
    for _ in range(num_of_events):
        found_event = False
        while not found_event:
            if counter >= retries:
                return benign_events
            try:
                cur_event = random.randint(2 * gap_days + 1, cur_repo_data.shape[0] - gap_days * 2 - 1)
            except ValueError:
                counter += 1
                continue
            event = cur_repo_data.index[cur_event]

            before_vuln = event - gap_days
            after_vuln = event + gap_days
            res_event = cur_repo_data[before_vuln:event - 1]
            if not res_event[res_event["VulnEvent"] > 0].empty:
                counter += 1
                continue
            benign_events.append(res_event.iloc[:, :-1].values)
            found_event = True

    return benign_events


def create_all_events(cur_repo_data, gap_days):
    all_events = []
    labels = []
    for i in range(gap_days, cur_repo_data.shape[0], 1):
        event = cur_repo_data.index[i]
        before_vuln = event - gap_days
        res_event = cur_repo_data[before_vuln:event - 1]
        all_events.append(res_event.iloc[:, :-1].values)
        labels.append(res_event.iloc[:, -1].values)
    return all_events, labels


def add_time_one_hot_encoding(df, with_idx=False):
    hour = pd.get_dummies(df.index.get_level_values(0).hour.astype(pd.CategoricalDtype(categories=range(24))),
                          prefix='hour')
    week = pd.get_dummies(df.index.get_level_values(0).day_of_week.astype(pd.CategoricalDtype(categories=range(7))),
                          prefix='day_of_week')
    day_of_month = pd.get_dummies(df.index.get_level_values(0).day.astype(pd.CategoricalDtype(categories=range(1, 32))),
                                  prefix='day_of_month')

    df = pd.concat([df.reset_index(), hour, week, day_of_month], axis=1)
    if with_idx:
        df = df.set_index(['created_at', 'idx'])
    else:
        df = df.set_index(['index'])
    return df


def get_event_window(cur_repo_data, event, aggr_options, days=10, hours=10, backs=50, resample=24):
    starting_time = event[0] - timedelta(days=days, hours=hours)
    res = cur_repo_data[starting_time:event[0]]

    if aggr_options == Aggregate.before_cve:
        res = res.iloc[:-1, :]
        res = res.reset_index().drop(["idx"], axis=1).set_index("created_at")
        res = res.resample(f'{resample}H').sum()
        res = add_time_one_hot_encoding(res, with_idx=False)

    elif aggr_options == Aggregate.after_cve:
        res = res.iloc[:-1, :]
        res = res.reset_index().drop(["idx"], axis=1).set_index("created_at")
        new_row = pd.DataFrame([[0] * len(res.columns)], columns=res.columns, index=[starting_time])
        res = pd.concat([new_row, res], ignore_index=False)
        res = res.resample(f'{resample}H').sum()
        res = add_time_one_hot_encoding(res, with_idx=False)

    else:
        res = cur_repo_data.reset_index().drop(["created_at"], axis=1).set_index("idx")[event[1] - backs:event[1]]
    return res.values


repo_dirs = '../repo_gharchive_processed4'
benign_all, vuln_all = [], []
n_features = 0
gap_days = 150

nice_list = ['facebook_hhvm.csv',
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
             # 'owncloud_core.csv',
             'php_php-src.csv',
             'revive-adserver_revive-adserver.csv',
             # 'rubygems_rubygems.csv',
             # 'the-tcpdump-group_tcpdump.csv'
             ]


class Aggregate(Enum):
    none = "none"
    before_cve = "before"
    after_cve = "after"


def create_dataset(aggr_options, benign_vuln_ratio, hours, days, resample, backs):
    vuln_all, benign_all = [], []
    vuln_details, benign_details = [], []
    for file in os.listdir(repo_dirs)[:]:
        try:
            selected = 4
            if file not in nice_list[:]:
                continue
            cur_repo = pd.read_csv(repo_dirs + "/" + file, parse_dates=['created_at'])
            cur_repo['idx'] = range(len(cur_repo))
            cur_repo = cur_repo.set_index(["created_at", "idx"])
            if cur_repo.shape[0] < 100:
                continue

            # cur_repo = cur_repo[cur_repo.index.notnull()]
            cur_repo["additions"] = (cur_repo["additions"] - cur_repo["additions"].mean()) / cur_repo["additions"].std()
            cur_repo["deletions"] = (cur_repo["deletions"] - cur_repo["deletions"].mean()) / cur_repo["deletions"].std()

        except pd.errors.EmptyDataError:
            continue

        cols_at_end = ['VulnEvent']
        cur_repo = cur_repo[[c for c in cur_repo if c not in cols_at_end]
                            + [c for c in cols_at_end if c in cur_repo]]

        vulns = cur_repo.index[cur_repo['VulnEvent'] > 0].tolist()
        benigns = cur_repo.index[cur_repo['VulnEvent'] == 0].tolist()
        random.shuffle(benigns)
        if aggr_options == Aggregate.none:
            cur_repo = add_time_one_hot_encoding(cur_repo, with_idx=True)

        for vuln in tqdm.tqdm(vulns, desc=file + " vuln", leave=False):
            res = get_event_window(cur_repo, vuln, aggr_options, days=days, hours=hours, backs=backs,
                                            resample=resample)
            tag = 1
            details = (file,vuln,tag)
            vuln_all.append(res)
            vuln_details.append(details)
        benign_counter = 0
        for benign in tqdm.tqdm(benigns, file + " benign", leave=False):
            if benign_counter >= benign_vuln_ratio * len(vulns):
                break

            res = get_event_window(cur_repo, benign, aggr_options, days=days, hours=hours, backs=backs,
                                            resample=resample)
            benign_all.append(res)
            tag = 0
            details = (file,vuln,tag)
            benign_details.append(details)
            benign_counter += 1

        # print(file, res.shape, to_padA, to_padB)

    # Padding
    padded_vuln_all, padded_benign_all = [], []
    to_pad = max(max(Counter([v.shape[0] for v in vuln_all])), max(Counter([v.shape[0] for v in benign_all])))

    for vuln in vuln_all:
        padded_vuln_all.append(np.pad(vuln, ((to_pad - vuln.shape[0], 0), (0, 0))))

    for benign in benign_all:
        padded_benign_all.append(np.pad(benign, ((to_pad - benign.shape[0], 0), (0, 0))))

    vuln_all = np.nan_to_num(np.array(padded_vuln_all))
    benign_all = np.nan_to_num(np.array(padded_benign_all))

    # Saving To File
    benign_npy_name, vuln_npy_name = make_file_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample)
    np.save("ready_data/" + vuln_npy_name, vuln_all)
    np.save("ready_data/" + benign_npy_name, benign_all)
    np.save("ready_data/details_" + vuln_npy_name, vuln_details)
    np.save("ready_data/details_" + benign_npy_name, benign_details)

    return vuln_all, benign_all, vuln_details, benign_details


def make_file_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample):
    name_template = f"{str(aggr_options)}_{benign_vuln_ratio}_H{hours}_D{days}_R{resample}_B{backs}"
    print(name_template)
    vuln_npy_name = name_template + "_vuln.npy"
    benign_npy_name = name_template + "_benign.npy"
    return benign_npy_name, vuln_npy_name


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hours', type=int, default=0, help='hours back')
    parser.add_argument('-d', '--days', type=int, default=10, help='days back')
    parser.add_argument('--resample', type=int, default=24, help='should resample aggregate')
    parser.add_argument('-r', '--ratio', type=int, default=1, help='benign vuln ratio')
    parser.add_argument('-a', '--aggr', type=Aggregate, action=EnumAction, default=Aggregate.none)
    parser.add_argument('-b', '--backs', type=int, default=10, help=' using none aggregation, operations back')
    parser.add_argument('-v', '--verbose', help="Be verbose", action="store_const", dest="loglevel", const=logging.INFO)
    args = parser.parse_args()
    return args


def extract_dataset(aggr_options=Aggregate.none, benign_vuln_ratio=1, hours=0, days=10, resample=12, backs=50):
    benign_npy_name, vuln_npy_name = make_file_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample)

    if os.path.isfile("ready_data/" + vuln_npy_name) and os.path.isfile('ready_data/' + benign_npy_name):
        logging.info(f"Loading Dataset {benign_npy_name}")
        vuln_all = np.load("ready_data/" + vuln_npy_name)
        benign_all = np.load("ready_data/" + benign_npy_name)
        vuln_details = np.load("ready_data/details_" + vuln_npy_name,allow_pickle=True)
        benign_details = np.load("ready_data/details_" + benign_npy_name, allow_pickle=True)
    else:
        logging.info(f"Creating Dataset {benign_npy_name}")
        vuln_all, benign_all, vuln_details, benign_details = create_dataset(aggr_options, benign_vuln_ratio, hours,
                                              days, resample, backs)

    all_train_x = np.concatenate([vuln_all, benign_all])
    all_train_y = np.concatenate([vuln_details, benign_details])

    return all_train_x, all_train_y


def train_model(X_train, y_train, X_test, y_test):
    part = 1
    X_train = X_train[:X_train.shape[0] // part, :, :]
    X_test = X_test[:X_test.shape[0] // part, :, :]
    y_train = y_train[:y_train.shape[0] // part]
    y_test = y_test[:y_test.shape[0] // part]

    model1 = Sequential()
    model1.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
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

    reshaped_train, reshaped_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)

    # define model
    model2 = Sequential()
    model2.add(Conv1D(filters=500, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model2.add(Dropout(0.4))
    model2.add(MaxPooling1D(pool_size=2))
    model2.add(Dropout(0.3))
    model2.add(Flatten())
    model2.add(Dense(100, activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Dense(100, activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Dense(100, activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Dense(70, activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Dense(50, activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Dense(30, activation='relu'))
    model2.add(Dropout(0.3))
    model2.add(Dense(1, activation='sigmoid'))
    model2.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=30)

    verbose = 0
    if logging.INFO <= logging.root.level:
        verbose = 1

    history = model2.fit(X_train, y_train[:,2], verbose=verbose, epochs=30, validation_data=(X_test, y_test[:,2]), callbacks=[])

    # print(model.evaluate(X_test.reshape(X_test.shape[0],-1), y_test, verbose=0))
    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['val_accuracy'])
    pyplot.title('model accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'val'], loc='upper left')
    pyplot.draw()
    return model2


def check_results(all_train_x, all_train_y, model):
    normalized_all_train_x = normalize(all_train_x)

    X_train, X_test, y_train, y_test = train_test_split(normalized_all_train_x, all_train_y, shuffle=True,
                                                        random_state=42)

    f1, thresh, best_y = find_best_f1(X_test, y_test, model)
    print(f1)

    pred = model.predict(X_test).reshape(-1)
    real = y_test
    df = DataFrame(zip(real, pred), columns=['real', 'pred', 'details'])
    fps = df[(df['pred'] > df['real']) & (df['pred'] > 0.5)].index.values
    for fp in fps:
        cur = df.iloc[fp]
        print(cur)


def main():
    args = parse_args()
    logging.basicConfig(level=args.loglevel)
    all_train_x, all_train_y = extract_dataset(aggr_options=args.aggr,
                                               resample=args.resample,
                                               benign_vuln_ratio=args.ratio,
                                               hours=args.hours,
                                               days=args.days,
                                               backs=args.backs)
    normalized_all_train_x = normalize(all_train_x)

    X_train, X_test, y_train, y_test = train_test_split(normalized_all_train_x, all_train_y, shuffle=True,
                                                        random_state=42)

    model = train_model(X_train, y_train, X_test, y_test)
    check_results(all_train_x, all_train_y, model)


if __name__ == '__main__':
    main()
