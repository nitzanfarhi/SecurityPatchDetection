#!/usr/bin/env python
# coding: utf-8
import json
import os
import logging
import pickle
import datetime as dt
import pandas as pd
import numpy as np
import pandas as pd
import tqdm
import random
import itertools

from datetime import datetime, timedelta
from dateutil import parser
from numpy import array

from collections import Counter
from pandas import DataFrame
from enum import Enum
from matplotlib import pyplot
from matplotlib import pyplot as plt
from classes import Repository

import helper
from helper import normalize, find_best_f1, find_best_accuracy, EnumAction,safe_mkdir

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import argparse
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot



def find_benign_events(cur_repo_data, gap_days, num_of_events):
    """
    :param cur_repo_data: DataFrame that is processed
    :param gap_days: number of days to look back for the events
    :param num_of_events: number of events to find
    :return: list of all events
    """
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
    """
    :param cur_repo_data: DataFrame that is processed
    :param gap_days: number of days to look back for the events
    :return: list of all events
    """
    all_events = []
    labels = []
    for i in range(gap_days, cur_repo_data.shape[0], 1):
        event = cur_repo_data.index[i]
        before_vuln = event - gap_days
        res_event = cur_repo_data[before_vuln:event - 1]
        all_events.append(res_event.iloc[:, :-1].values)
        labels.append(res_event.iloc[:, -1].values)
    return all_events, labels


event_types = ['PullRequestEvent', 'PushEvent', 'ReleaseEvent', 'DeleteEvent', 'issues', 'CreateEvent', 'releases', 'IssuesEvent', 'ForkEvent', 'WatchEvent', 'PullRequestReviewCommentEvent', 'stargazers', 'pullRequests', 'commits', 'CommitCommentEvent', 'MemberEvent', 'GollumEvent', 'IssueCommentEvent', 'forks', 'PullRequestReviewEvent', 'PublicEvent', 'VulnEvent']


def add_type_one_hot_encoding(df):
    """
    :param df: dataframe to add type one hot encoding to
    :return: dataframe with type one hot encoding
    """
    type_one_hot = pd.get_dummies(df.type.astype(pd.CategoricalDtype(categories=event_types)))
    df = pd.concat([df, type_one_hot], axis=1)
    return df

def add_time_one_hot_encoding(df, with_idx=False):
    """
    :param df: dataframe to add time one hot encoding to
    :param with_idx: if true, adds index column to the dataframe
    :return: dataframe with time one hot encoding
    """

    hour = pd.get_dummies(df.index.get_level_values(0).hour.astype(pd.CategoricalDtype(categories=range(24))),
                          prefix='hour')
    week = pd.get_dummies(df.index.get_level_values(0).dayofweek.astype(pd.CategoricalDtype(categories=range(7))),
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
    """
    :param cur_repo_data: DataFrame that is processed
    :param event: list of events to get windows from
    :param aggr_options: can be before, after or none, to decide how we agregate
    :param days: if 'before' or 'after' is choosed as aggr_options
        amount of days gathered as a single window (in addition to hours)
    :param hours: if 'before' or 'after' is choosed as aggr_options
        amount of hours gathered as a single window (in addition to days)
    :param backs: if 'none' is choosed as aggr_options, this is the amount of events back taken
    :param resample: is the data resampled and at what frequency (hours)
    :return: a window for lstm
    """
    befs = -10
    if aggr_options == Aggregate.after_cve:
        cur_repo_data = cur_repo_data.reset_index().drop(["idx"], axis=1).set_index("created_at")
        cur_repo_data = cur_repo_data.sort_index()
        starting_time = event[0] - timedelta(days=days, hours=hours)
        res = cur_repo_data[starting_time:event[0]]
        res = res.iloc[:befs, :]
        new_row = pd.DataFrame([[0] * len(res.columns)], columns=res.columns, index=[starting_time])
        res = pd.concat([new_row, res], ignore_index=False)
        res = res.resample(f'{resample}H').sum()
        res = add_time_one_hot_encoding(res, with_idx=False)

    elif aggr_options == Aggregate.none:
        res = cur_repo_data.reset_index().drop(["created_at"], axis=1).set_index("idx")[
              event[1] - backs:event[1] + befs]
    return res.values


repo_dirs = 'hiddenCVE/gh_cve_proccessed'
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

nice_list2 = ['abrt_abrt.csv', 'clusterlabs_pcs.csv', 'discourse_discourse.csv', 'exponentcms_exponent-cms.csv',
              'facebook_hhvm.csv', 'ffmpeg_ffmpeg.csv', 'file_file.csv', 'firefly-iii_firefly-iii.csv',
              'flatpak_flatpak.csv', 'freerdp_freerdp.csv', 'fusionpbx_fusionpbx.csv', 'git_git.csv', 'gpac_gpac.csv',
              'ifmeorg_ifme.csv', 'imagemagick_imagemagick.csv', 'jenkinsci_jenkins.csv', 'kanboard_kanboard.csv',
              'kde_kdeconnect-kde.csv', 'koral--_android-gif-drawable.csv', 'krb5_krb5.csv',
              'libarchive_libarchive.csv', 'libgit2_libgit2.csv', 'libraw_libraw.csv',
              'livehelperchat_livehelperchat.csv', 'mantisbt_mantisbt.csv', 'mdadams_jasper.csv', 'oisf_suricata.csv',
              'op-tee_optee_os.csv', 'openssh_openssh-portable.csv', 'openssl_openssl.csv', 'owncloud_core.csv',
              'php_php-src.csv']


class Aggregate(Enum):
    none = "none"
    before_cve = "before"
    after_cve = "after"

def create_dataset(aggr_options, benign_vuln_ratio, hours, days, resample, backs):
    """

    :param aggr_options: can be before, after or none, to decide how we agregate
    :param benign_vuln_ratio: ratio of benign to vuln
    :param hours: if 'before' or 'after' is choosed as aggr_options
    :param days:    if 'before' or 'after' is choosed as aggr_options
    :param resample: is the data resampled and at what frequency (hours)
    :param backs: if 'none' is choosed as aggr_options, this is the amount of events back taken
    :return: dataset
    """
    all_repos = []
    all_set = set()

    dirname = make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample)
    safe_mkdir("ready_data")
    safe_mkdir("ready_data/" + dirname)

    for file in os.listdir(repo_dirs):
        repo_holder = Repository()
        repo_holder.file = file
        try:
            cur_repo = pd.read_csv(repo_dirs + "/" + file, parse_dates=['created_at'])
        except pd.errors.EmptyDataError:
            continue

        cur_repo = cur_repo.sort_index()
        cur_repo = cur_repo[cur_repo["created_at"].notnull()]
        all_set.update(cur_repo.type.unique())
        cur_repo['idx'] = range(len(cur_repo))
        cur_repo = cur_repo.set_index(["created_at", "idx"])
        if cur_repo.shape[0] < 100:
            continue

        # cur_repo = cur_repo[cur_repo.index.notnull()]
        for commit_change in ["additions", "deletions"]:
            cur_repo[commit_change].fillna(0, inplace=True)
            cur_repo[commit_change] = cur_repo[commit_change].astype(int)
            cur_repo[commit_change] = (cur_repo[commit_change] - cur_repo[commit_change].mean()) / cur_repo[commit_change].std()



        cur_repo["is_vuln"] = cur_repo.type.apply(lambda x: 1 if x == "VulnEvent" else 0)

        cur_repo = add_type_one_hot_encoding(cur_repo)
        cur_repo = cur_repo.drop(["type"], axis=1)
        cur_repo = cur_repo.drop(["name"], axis=1)
        cur_repo = cur_repo.drop(["Unnamed: 0"],axis=1)
        vulns = cur_repo.index[cur_repo['is_vuln'] > 0].tolist()
        if len(vulns) < 10:
            continue

        print(file, ",", len(vulns))

        benigns = cur_repo.index[cur_repo['is_vuln'] == 0].tolist()
        random.shuffle(benigns)

        cols_at_end = ['is_vuln']
        cur_repo = cur_repo[[c for c in cur_repo if c not in cols_at_end]
                            + [c for c in cols_at_end if c in cur_repo]]

        if aggr_options == Aggregate.none:
            cur_repo = add_time_one_hot_encoding(cur_repo, with_idx=True)

        for vuln in tqdm.tqdm(vulns, desc=file + " vuln", leave=False):
            res = get_event_window(cur_repo, vuln, aggr_options, days=days, hours=hours, backs=backs,
                                   resample=resample)
            tag = 1
            details = (file, vuln, tag)
            repo_holder.vuln_lst.append(res)
            repo_holder.vuln_details.append(details)

        for benign in tqdm.tqdm(benigns[:benign_vuln_ratio*len(vulns)], file + " benign", leave=False):

            res = get_event_window(cur_repo, benign, aggr_options, days=days, hours=hours, backs=backs,
                                   resample=resample)
            tag = 0
            details = (file, benign, tag)
            repo_holder.benign_lst.append(res)
            repo_holder.benign_details.append(details)

        repo_holder.pad_repo()
        with open("ready_data/" + dirname + "/" + repo_holder.file + ".pkl", 'wb') as f:
            pickle.dump(repo_holder, f)
        all_repos.append(repo_holder)
    print(all_set)
    return all_repos


def make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample):
    """
    :return: name of the directory to save the data in
    """
    name_template = f"{str(aggr_options)}_{benign_vuln_ratio}_H{hours}_D{days}_R{resample}_B{backs}"
    print(name_template)
    return name_template


def extract_dataset(aggr_options=Aggregate.none, benign_vuln_ratio=1, hours=0, days=10, resample=12, backs=50,
                    cache=False):
    """
    :param aggr_options: Aggregate.none, Aggregate.before_cve, Aggregate.after_cve
    :param benign_vuln_ratio: ratio of benign to vuln events
    :param hours: hours before and after vuln event
    :param days: days before and after vuln event
    :param resample: resample window
    :param backs: number of backs to use
    :param cache: if true, will use cached data
    :return: a list of Repository objects and dir name
    """

    dirname = make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample)
    if cache and os.path.isdir("ready_data/" + dirname) and len(os.listdir("ready_data/" + dirname)) != 0:
        logging.info(f"Loading Dataset {dirname}")
        all_repos = []
        for file in os.listdir("ready_data/" + dirname):
            with open("ready_data/" + dirname + "/" + file, 'rb') as f:
                repo = pickle.load(f)
                all_repos.append(repo)

    else:
        logging.info(f"Creating Dataset {dirname}")
        all_repos = create_dataset(aggr_options, benign_vuln_ratio, hours, days, resample, backs)

    return all_repos, dirname


def evaluate_data(X_train, y_train,X_val,y_val, X_test, y_test, exp_name, epochs=20, fp=False):

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
    from tensorflow.keras import Input, layers
    from tensorflow.keras.callbacks import EarlyStopping


    """
    Evaluate the model with the given data.
    """
    # X_train = X_train[:X_train.shape[0] // part, :, :]
    # X_test = X_test[:X_test.shape[0] // part, :, :]
    # y_train = y_train[:y_train.shape[0] // part]
    # y_test = y_test[:y_test.shape[0] // part]

    used_y_train = np.asarray(y_train).astype('float32')
    used_y_test = np.asarray(y_test).astype('float32')

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
    model2.add(Dropout(0.1))
    model2.add(MaxPooling1D(pool_size=2))
    model2.add(Dropout(0.1))
    model2.add(Flatten())
    model2.add(Dense(100, activation='relu'))
    model2.add(Dropout(0.1))
    model2.add(Dense(100, activation='relu'))
    model2.add(Dropout(0.1))
    model2.add(Dense(100, activation='relu'))
    model2.add(Dropout(0.1))
    model2.add(Dense(70, activation='relu'))
    model2.add(Dropout(0.1))
    model2.add(Dense(50, activation='relu'))
    model2.add(Dropout(0.1))
    model2.add(Dense(30, activation='relu'))
    model2.add(Dropout(0.1))
    model2.add(Dense(1, activation='sigmoid'))
    model2.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                   metrics=['accuracy'])

    model3 = Sequential()
    model3.add(LSTM(units=100, activation='relu', name='first_lstm', recurrent_dropout=0.1,
                    input_shape=(X_train.shape[1], X_train.shape[2])))
    model3.add(Dense(1, activation="sigmoid"))

    model3.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                   metrics=['accuracy'])

    model4 = Sequential()
    model4.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model4.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model4.add(Dropout(0.5))
    model4.add(MaxPooling1D(pool_size=2))
    model4.add(Flatten())
    model4.add(Dense(100, activation='relu'))
    model4.add(Dense(1, activation='sigmoid'))
    model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=50)

    verbose = 0
    if logging.INFO <= logging.root.level:
        verbose = 1

    model = model4
    history = model.fit(X_train, used_y_train, verbose=verbose, epochs=epochs, shuffle=True,
                        validation_data=(X_test, used_y_test), callbacks=[es])

    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['val_accuracy'])
    pyplot.title('model accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'val'], loc='upper left')
    pyplot.draw()

    safe_mkdir("figs")
    pyplot.savefig(f"figs/{exp_name}_{epochs}.png")

    # Final evaluation of the model
    pred = model.predict(X_test).reshape(-1)

    accuracy = check_results(X_test, y_test, pred, model, exp_name, fp=fp)

    return accuracy


def acquire_commits(name, date):
    """
    Acquire the commits for the given repository.
    """
    group, repo = name.replace(".csv", "").split("_", 1)

    github_format = "%Y-%m-%dT00:00:00"
    for branch in ["master", "main"]:
        res = helper.run_query(
            helper.commits_between_dates.format(group,
                                                repo,
                                                branch,
                                                date.strftime(github_format),
                                                (date + timedelta(days=1)).strftime(github_format)
                                                ))
        if "data" in res:
            if "repository" in res["data"]:
                if "object" in res['data']['repository']:
                    obj = res['data']['repository']['object']
                    if obj is None:
                        continue
                    if "history" in obj:
                        return res['data']['repository']['object']['history']['nodes']
    return ""


def check_results(X_test, y_test, pred, model, exp_name, fp=False):
    """
    Check the results of the model.
    """
    used_y_test = np.asarray(y_test).astype('float32')
    scores = model.evaluate(X_test, used_y_test, verbose=0)
    max_f1, thresh, _ = find_best_f1(X_test, used_y_test, model)
    print(max_f1, thresh)
    with open(f"results/{exp_name}.txt", 'w') as mfile:
        mfile.write('Accuracy: %.2f%%\n' % (scores[1] * 100))
        mfile.write('fscore: %.2f%%\n' % (max_f1 * 100))

        print('Accuracy: %.2f%%' % (scores[1] * 100))
        print('fscore: %.2f%%' % (max_f1 * 100))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(used_y_test, pred)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2

    plt.plot(fpr['micro'], tpr['micro'], color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc['micro'])
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()

    if fp:
        real = y_test[:, 2]
        date = y_test[:, 1]
        name = y_test[:, 0]
        df = DataFrame(zip(real, pred, date, name), columns=['real', 'pred', 'date', 'name'])
        fps = df[(df['pred'] > df['real']) & (df['pred'] > 0.5)]
        for index, row in tqdm.tqdm(list(fps.iterrows())):
            with open(f'output/{row["name"]}_{row["date"][0].strftime("%Y-%m-%d")}_{str(row["date"][1])}.json',
                      'w+') as mfile:
                commits = acquire_commits(row["name"], row["date"][0])
                json.dump(commits, mfile, indent=4, sort_keys=True)
    return scores[1]


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hours', type=int, default=0, help='hours back')
    parser.add_argument('-d', '--days', type=int, default=10, help='days back')
    parser.add_argument('--resample', type=int, default=24, help='number of hours that should resample aggregate')
    parser.add_argument('-r', '--ratio', type=int, default=1, help='benign vuln ratio')
    parser.add_argument('-a', '--aggr', type=Aggregate, action=EnumAction, default=Aggregate.none)
    parser.add_argument('-b', '--backs', type=int, default=10, help=' using none aggregation, operations back')
    parser.add_argument('-v', '--verbose', help="Be verbose", action="store_const", dest="loglevel", const=logging.INFO)
    parser.add_argument('-c', '--cache', '--cached', help="Use Cached Data", action="store_const", dest="cache",  const=True)
    parser.add_argument('-e', '--epochs', type=int, default=10, help=' using none aggregation, operations back')
    parser.add_argument('-f', '--find-fp', help="Find False positive commits", action="store_const",
                        dest="fp", const=True)

    args = parser.parse_args()
    return args


def split_into_x_and_y(repos):
    """
    Split the repos into X and Y.
    """
    X_train, y_train = [], []
    for repo in repos:
        x, y = repo.get_all_lst()
        X_train.append(x)
        y_train.append(y)
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    return X_train, y_train


def main():
    args = parse_args()
    logging.basicConfig(level=args.loglevel)
    all_repos, exp_name = extract_dataset(aggr_options=args.aggr,
                                          resample=args.resample,
                                          benign_vuln_ratio=args.ratio,
                                          hours=args.hours,
                                          days=args.days,
                                          backs=args.backs,
                                          cache=args.cache)

    to_pad = max([x.get_all_lst()[0].shape[1] for x in all_repos])
    for repo in all_repos:
        repo.pad_repo(to_pad)

    train_size = int(0.7 * len(all_repos))
    validation_size = int(0.15 * len(all_repos))
    test_size = int(0.15 * len(all_repos))

    train_repos = all_repos[:train_size]
    validation_repos = all_repos[train_size:train_size+validation_size]
    test_repos = all_repos[train_size+validation_size:]

    X_train, y_train = split_into_x_and_y(train_repos)
    X_val, y_val = split_into_x_and_y(validation_repos)
    X_test, y_test = split_into_x_and_y(test_repos)

    res = evaluate_data(X_train, y_train, X_val,y_val, X_test, y_test, exp_name, epochs=args.epochs, fp=args.fp)
    print(res)


if __name__ == '__main__':
    main()
