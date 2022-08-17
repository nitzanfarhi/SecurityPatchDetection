#!/usr/bin/env python
# coding: utf-8
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD

from models import *
from sklearn.metrics import roc_curve, auc, confusion_matrix
from helper import find_best_f1, EnumAction, safe_mkdir
from classes import Repository
from matplotlib import pyplot as plt
from matplotlib import pyplot
from enum import Enum
from pandas import DataFrame
from keras_tuner import RandomSearch
from hiddenCVE.graphql import all_langs
from dateutil import parser

import helper
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import matplotlib
import datetime
import random
import tqdm
import pickle
import json
import os
import logging

import coloredlogs
import logging
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


logger = logging.getLogger(__name__)
coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s')


matplotlib.use('Agg')


BENIGN_TAG = 0
VULN_TAG = 1


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
                cur_event = random.randint(
                    2 * gap_days + 1, cur_repo_data.shape[0] - gap_days * 2 - 1)
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


event_types = ['PullRequestEvent', 'PushEvent', 'ReleaseEvent', 'DeleteEvent', 'issues', 'CreateEvent', 'releases', 'IssuesEvent', 'ForkEvent', 'WatchEvent', 'PullRequestReviewCommentEvent',
               'stargazers', 'pullRequests', 'commits', 'CommitCommentEvent', 'MemberEvent', 'GollumEvent', 'IssueCommentEvent', 'forks', 'PullRequestReviewEvent', 'PublicEvent']


def add_type_one_hot_encoding(df):
    """
    :param df: dataframe to add type one hot encoding to
    :return: dataframe with type one hot encoding
    """
    type_one_hot = pd.get_dummies(df.type.astype(
        pd.CategoricalDtype(categories=event_types)))
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
    befs = -1
    hours_befs = 2

    if aggr_options == Aggregate.after_cve:
        cur_repo_data = cur_repo_data.reset_index().drop(
            ["idx"], axis=1).set_index("created_at")
        cur_repo_data = cur_repo_data.sort_index()
        indicator = event[0] - datetime.timedelta(days=0, hours=hours_befs)
        starting_time = indicator - datetime.timedelta(days=days, hours=hours)
        res = cur_repo_data[starting_time:indicator]
        new_row = pd.DataFrame([[0] * len(res.columns)],
                               columns=res.columns, index=[starting_time])
        res = pd.concat([new_row, res], ignore_index=False)
        res = res.resample(f'{resample}H').sum()
        res = add_time_one_hot_encoding(res, with_idx=False)

    elif aggr_options == Aggregate.before_cve:
        res = cur_repo_data.reset_index().drop(["created_at"], axis=1).set_index("idx")[
            event[1] - backs:event[1] + backs]

    elif aggr_options == Aggregate.none:
        res = cur_repo_data.reset_index().drop(["created_at"], axis=1).set_index("idx")[
            event[1] - backs:event[1] + befs]
    return res.values


repo_dirs = 'hiddenCVE/gh_cve_proccessed'
repo_metadata = 'hiddenCVE/repo_metadata.json'
benign_all, vuln_all = [], []
n_features = 0
gap_days = 150


class Aggregate(Enum):
    none = "none"
    before_cve = "before"
    after_cve = "after"


def create_dataset(aggr_options, benign_vuln_ratio, hours, days, resample, backs, metadata=False):
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
    ignored = []
    dirname = make_new_dir_name(
        aggr_options, backs, benign_vuln_ratio, days, hours, resample)
    safe_mkdir("ready_data")
    safe_mkdir("ready_data/" + dirname)

    with open(repo_metadata,'r') as mfile:
        all_metadata = json.load(mfile)

    counter = 0
    for file in (pbar := tqdm.tqdm(os.listdir(repo_dirs))):
        if ".csv" not in file:
            continue
        file = file.split(".csv")[0]
        counter += 1
        # if counter > 20:
        #     break
        repo_holder = Repository()
        repo_holder.file = file
        pbar.set_description(f"{file} read")

        if not os.path.exists(repo_dirs+"/"+file+".parquet"):
            try:
                cur_repo = pd.read_csv(
                    repo_dirs + "/" + file+".csv",
                    low_memory=False,
                    parse_dates=['created_at'],
                    dtype={"type": "string", "name": "string", "Hash": "string", "Add":  np.float64, "Del":  np.float64, "Files": np.float64, "Vuln":  np.float64})
            except pd.errors.EmptyDataError:
                continue
            
            cur_repo.to_parquet(repo_dirs + "/" + file+".parquet")

        else:
            cur_repo = pd.read_parquet(repo_dirs + "/" + file+".parquet")

        if cur_repo.shape[0] < 100:
            ignored.append(file)
            continue
        cur_repo["Hash"] = cur_repo["Hash"].fillna("")
        cur_repo = cur_repo.fillna(0)

        number_of_vulns = cur_repo[cur_repo['Vuln'] != 0].shape[0]
        if number_of_vulns == 0:
            ignored.append((file, number_of_vulns))
            continue

        pbar.set_description(f"{file},{number_of_vulns} fix_repo_shape")

        cur_repo = fix_repo_shape(all_set, cur_repo)


        if metadata:
            cur_metadata = all_metadata[file.replace("_","/",1)]

            for key,value in cur_metadata.items():
                if key == "languages_edges":
                    for lang in all_langs:
                        cur_repo[lang] = 0
                    for lang in value:
                        cur_repo[lang] = 1
                    continue
                if key == 'primaryLanguage_name':
                    continue
                if key == "createdAt":
                    cur_repo["repo_creation_data"]=parser.parse(value).year
                    continue
                if key == "fundingLinks":
                    cur_repo[key]=len(value)
                    continue
                cur_repo[key]=value



        vulns = cur_repo.index[cur_repo['Vuln'] == 1].tolist()
        if not len(vulns):
            continue
        benigns = cur_repo.index[cur_repo['Vuln'] == 0].tolist()
        random.shuffle(benigns)
        benigns = benigns[:benign_vuln_ratio*len(vulns)]

        cur_repo = cur_repo.drop(["Vuln"], axis=1)
        pbar.set_description(f"{file} extract_window")
        if aggr_options == Aggregate.none:
            cur_repo = add_time_one_hot_encoding(cur_repo, with_idx=True)

        extract_window(aggr_options, hours, days, resample, backs,
                       file, repo_holder.vuln_lst, repo_holder.vuln_details, cur_repo, vulns, VULN_TAG)

        extract_window(aggr_options, hours, days, resample, backs,
                       file, repo_holder.benign_lst, repo_holder.benign_details, cur_repo, benigns, BENIGN_TAG)

        pbar.set_description(f"{file} pad")

        repo_holder.pad_repo()

        pbar.set_description(f"{file} save")

        with open("ready_data/" + dirname + "/" + repo_holder.file + ".pkl", 'wb') as f:
            pickle.dump(repo_holder, f)

        all_repos.append(repo_holder)

    return all_repos


def extract_window(aggr_options, hours, days, resample, backs, file, window_lst, details_lst, cur_repo, cur_list, tag):
    """
    pulls out a window of events from the repo
    :param aggr_options: can be before, after or none, to decide how we agregate
    :param hours: if 'before' or 'after' is choosed as aggr_options
    :param days:    if 'before' or 'after' is choosed as aggr_options
    :param resample: is the data resampled and at what frequency (hours)
    :param backs: if 'none' is choosed as aggr_options, this is the amount of events back taken
    :param file: the file name
    :param repo_holder: the repo holder
    :param cur_repo: the current repo
    :param cur_list: the current list of events
    :param tag: the tag to add to the window
    :return: None
    """
    for cur in cur_list:
        res = get_event_window(cur_repo, cur, aggr_options,
                               days=days, hours=hours, backs=backs, resample=resample)
        details = (file, cur, tag)
        window_lst.append(res)
        details_lst.append(details)


def fix_repo_shape(all_set, cur_repo):
    cur_repo['created_at'] = pd.to_datetime(cur_repo['created_at'], utc=True)
    cur_repo = cur_repo[~cur_repo.duplicated(
        subset=['created_at', 'Vuln'], keep='first')]
    cur_repo = cur_repo.set_index(["created_at"])
    cur_repo = cur_repo.sort_index()
    cur_repo = cur_repo[cur_repo.index.notnull()]
    all_set.update(cur_repo.type.unique())
    cur_repo['idx'] = range(len(cur_repo))
    cur_repo = cur_repo.reset_index().set_index(["created_at", "idx"])

    # cur_repo = cur_repo[cur_repo.index.notnull()]
    for commit_change in ["Add", "Del", "Files"]:
        cur_repo[commit_change].fillna(0, inplace=True)
        cur_repo[commit_change] = cur_repo[commit_change].astype(int)
        cur_repo[commit_change] = (
            cur_repo[commit_change] - cur_repo[commit_change].mean()) / cur_repo[commit_change].std()

    cur_repo = add_type_one_hot_encoding(cur_repo)
    cur_repo = cur_repo.drop(["type"], axis=1)
    cur_repo = cur_repo.drop(["name"], axis=1)
    cur_repo = cur_repo.drop(["Unnamed: 0"], axis=1)
    cur_repo = cur_repo.drop(["Hash"], axis=1)
    return cur_repo


def make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample):
    """
    :return: name of the directory to save the data in
    """
    if aggr_options == Aggregate.before_cve:
        name_template = f"{str(aggr_options)}_R{benign_vuln_ratio}_B{backs}"
    elif aggr_options == Aggregate.after_cve:
        name_template = f"{str(aggr_options)}_R{benign_vuln_ratio}_RE{resample}_H{hours}_D{days}"
    elif aggr_options == Aggregate.none:
        name_template = f"{str(aggr_options)}_R{benign_vuln_ratio}_B{backs}"
    else:
        raise Exception("Aggr options not supported")
    logger.debug(name_template)
    return name_template


def extract_dataset(aggr_options=Aggregate.none, benign_vuln_ratio=1, hours=0, days=10, resample=12, backs=50,
                    cache=False,metadata=False):
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

    dirname = make_new_dir_name(
        aggr_options, backs, benign_vuln_ratio, days, hours, resample)
    if cache and os.path.isdir("ready_data/" + dirname) and len(os.listdir("ready_data/" + dirname)) != 0:
        logger.info(f"Loading Dataset {dirname}")
        all_repos = []
        for file in os.listdir("ready_data/" + dirname):
            with open("ready_data/" + dirname + "/" + file, 'rb') as f:
                repo = pickle.load(f)
                all_repos.append(repo)

    else:
        logger.info(f"Creating Dataset {dirname}")
        all_repos = create_dataset(
            aggr_options, benign_vuln_ratio, hours, days, resample, backs,metadata=metadata)

    return all_repos, dirname


def model_selector(model_name, shape1, shape2, optimizer):
    if model_name == "lstm":
        return lstm(shape1, shape2, optimizer)
    if model_name == "conv1d":
        return conv1d(shape1, shape2, optimizer)

    if model_name == "lstm_autoencoder":
        return lstm_autoencoder(shape1, shape2, optimizer)
    if model_name == "bilstm":
        return bilstm(shape1, shape2, optimizer)
    if model_name == "bigru":
        return bigru(shape1, shape2, optimizer)

    raise Exception("Model Not Found!")


def evaluate_data(X_train, y_train, X_val, y_val, exp_name, batch_size=32,  epochs=20, model_name="LSTM"):
    """
    Evaluate the model with the given data.
    """

    optimizer = SGD(learning_rate=0.1, momentum=0.9,
                    nesterov=True, clipnorm=1.)
    # Create the model
    model = model_selector(
        model_name, X_train.shape[1], X_train.shape[2], optimizer)
    safe_mkdir("models/")
    safe_mkdir("models/" + exp_name)
    mc = ModelCheckpoint(
        f'models/{exp_name}/best_model.h5', monitor='val_accuracy', mode='min', verbose=1)
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)

    verbose = 0
    if logger.level < logging.CRITICAL:
        verbose = 1

    validation_data = (X_val, y_val) if len(X_val) else None
    history = model.fit(X_train, y_train, verbose=verbose, epochs=epochs, shuffle=True, batch_size=batch_size,
                        validation_data=validation_data, callbacks=[es, mc])

    pyplot.plot(history.history['accuracy'])
    pyplot.plot(history.history['val_accuracy'])
    pyplot.title('model accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'val'], loc='upper left')

    safe_mkdir("figs")
    last_training = history.history['accuracy'][-1]
    last_validation = history.history['val_accuracy'][-1]
    pyplot.savefig(
        f"figs/{exp_name}_{epochs}_{model_name}_t{last_training}_v{last_validation}.png")

    # Final evaluation of the model
    return model


def acquire_commits(name, date, ignore_errors=False):
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
                                                (date + datetime.timedelta(days=1)
                                                 ).strftime(github_format)
                                                ), ignore_errors=ignore_errors)
        if "data" in res:
            if "repository" in res["data"]:
                if "object" in res['data']['repository']:
                    obj = res['data']['repository']['object']
                    if obj is None:
                        continue
                    if "history" in obj:
                        return res['data']['repository']['object']['history']['nodes']
    return ""


def check_results(X_test, y_test, pred, model, exp_name, model_name):
    """
    Check the results of the model.
    """
    used_y_test = np.asarray(y_test).astype('float32')
    scores = model.evaluate(X_test, used_y_test, verbose=0)
    max_f1, thresh, _ = find_best_f1(X_test, used_y_test, model)
    logger.debug(max_f1, thresh)
    with open(f"results/{exp_name}_{model_name}.txt", 'w') as mfile:
        mfile.write('Accuracy: %.2f%%\n' % (scores[1] * 100))
        mfile.write('fscore: %.2f%%\n' % (max_f1 * 100))
        mfile.write('confusion matrix:\n')
        tn, fp, fn, tp = confusion_matrix(y_test, pred > thresh).ravel()
        conf_matrix = f"tn={tn}, fp={fp}, fn={fn}, tp={tp}"
        mfile.write(conf_matrix)

        logger.critical('Accuracy: %.2f%%' % (scores[1] * 100))
        logger.critical('fscore: %.2f%%' % (max_f1 * 100))
        logger.critical(str(conf_matrix))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(used_y_test, pred)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2

    plt.plot(fpr['micro'], tpr['micro'], color="darkorange", lw=lw,
             label="ROC curve (area = %0.2f)" % roc_auc['micro'])
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig(f"figs/auc_{exp_name}_{roc_auc['micro']}.png")

    return scores[1]


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hours', type=int, default=0, help='hours back')
    parser.add_argument('-d', '--days', type=int, default=10, help='days back')
    parser.add_argument('--resample', type=int, default=24,
                        help='number of hours that should resample aggregate')
    parser.add_argument('-r', '--ratio', type=int,
                        default=1, help='benign vuln ratio')
    parser.add_argument('-a', '--aggr', type=Aggregate,
                        action=EnumAction, default=Aggregate.none)
    parser.add_argument('-b', '--backs', type=int, default=10,
                        help=' using none aggregation, operations back')
    parser.add_argument('-v', '--verbose', help="Be verbose",
                        action="store_const", dest="loglevel", const=logging.DEBUG)
    parser.add_argument('-c', '--cache', '--cached', help="Use Cached Data",
                        action="store_const", dest="cache",  const=True)
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help=' using none aggregation, operations back')
    parser.add_argument('-f', '--find-fp', help="Find False positive commits", action="store_const",
                        dest="fp", const=True)
    parser.add_argument('-m', '--model', action='store',
                        type=str, help='The model to receive.')

    parser.add_argument('-k', '--kfold', type=int,
                        default=10, help="Kfold cross validation")
    parser.add_argument('--batch', type=int, default=64, help="Batch size")
    parser.add_argument('--metadata',action="store_true", help="Use metadata")
    args = parser.parse_args()
    return args


def split_into_x_and_y(repos, with_details=False):
    """
    Split the repos into X and Y.
    """
    X_train, y_train = [], []
    details = []
    for repo in repos:
        x, y = repo.get_all_lst()
        if with_details:
            details.append(repo.get_all_details())
        X_train.append(x)
        y_train.append(y)
    if X_train:
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)

    if with_details:
        if details:
            details = np.concatenate(details)
        return X_train, y_train, details

    return X_train, y_train


def hypertune(X_train,y_train, X_test, y_test):
    tuner = RandomSearch(hypertune_bilstm(X_train.shape[1], X_train.shape[2]),
                         objective='val_accuracy',
                         max_trials=10,
                         executions_per_trial=3,
                         directory='tuner1',
                         project_name='Clothing2')
                         
    # es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20)

    tuner.search(X_train,y_train,epochs=100,validation_data=(X_test,y_test),callbacks=[])
    tuner.results_summary()

    print(tuner.get_best_hyperparameters(1))


def init():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = '0'


def main():
    args = parse_args()
    if not args.loglevel:
        logger.level = logging.CRITICAL
    else:
        logger.level = args.loglevel

    init()

    all_repos, exp_name = extract_dataset(aggr_options=args.aggr,
                                          resample=args.resample,
                                          benign_vuln_ratio=args.ratio,
                                          hours=args.hours,
                                          days=args.days,
                                          backs=args.backs,
                                          cache=args.cache,
                                          metadata = args.metadata)

    to_pad = 0
    num_of_vulns = 0
    for repo in all_repos:
        num_of_vulns += repo.get_num_of_vuln()
        if len(repo.get_all_lst()[0].shape) > 1:
            to_pad = max(to_pad, repo.get_all_lst()[0].shape[1])
        else:
            all_repos.remove(repo)

    for repo in all_repos:
        repo.pad_repo(to_pad=to_pad)
    TRAIN_SIZE = 0.8
    VALIDATION_SIZE = 0.0
    train_size = int(TRAIN_SIZE * num_of_vulns)
    validation_size = int(VALIDATION_SIZE * num_of_vulns)
    test_size = num_of_vulns - train_size - validation_size

    logger.info(f"Train size: {train_size}")
    logger.info(f"Validation size: {validation_size}")
    logger.info(f"Test size: {test_size}")

    accuracies = []
    for i in range(args.kfold):
        train_repos = []
        validation_repos = []
        test_repos = []

        vuln_counter = 0
        random.shuffle(all_repos)
        for repo in all_repos:
            if(vuln_counter < validation_size):
                logger.debug(f"Train - {repo.file}")
                validation_repos.append(repo)
            elif(vuln_counter < train_size + validation_size):
                logger.debug(f"Val - {repo.file}")
                train_repos.append(repo)
            else:
                logger.debug(f"Test - {repo.file}")
                test_repos.append(repo)
            vuln_counter += repo.get_num_of_vuln()

        if not train_repos or (not validation_repos and VALIDATION_SIZE != 0) or not test_repos:
            raise Exception("Not enough data, or data not splitted well")

        X_train, y_train = split_into_x_and_y(train_repos)
        X_val, y_val, val_details = split_into_x_and_y(
            validation_repos, with_details=True)
        X_test, y_test, test_details = split_into_x_and_y(
            test_repos, with_details=True)

        # X_test, y_test, X_val,y_val = X_val, y_val, X_test, y_test

        hypertune(X_train,y_train, X_test, y_test)
        return
        model = evaluate_data(X_train, y_train, X_test, y_test,
                              exp_name, batch_size=args.batch, epochs=args.epochs, model_name=args.model)

        pred = model.predict(X_test).reshape(-1)

        acc = check_results(X_test, y_test, pred, model, exp_name, args.model)

        accuracies.append(acc)

    print(f"Average accuracy: {np.mean(accuracies)}")
    with open(f"results/{exp_name}_K{args.kfold}.txt", 'w') as mfile:
        mfile.write(f"Average accuracy: {np.mean(accuracies)}")

    if args.fp:
        safe_mkdir("output")
        safe_mkdir(f"output/{exp_name}")

        summary_md = ""
        summary_md += f"# {exp_name}\n"

        df = DataFrame(zip(y_test, pred, test_details[:, 0], test_details[:, 1]), columns=[
                       'real', 'pred', 'file', 'timestamp'])

        fps = df[((df['pred'] > df['real']) & (df['pred'] > 0.5))]

        groups = fps.groupby('file')
        for name, group in groups:
            summary_md += f"## {name}\n"
            summary_md += "\n".join(
                list(group['timestamp'].apply(lambda x: "* " + str(x[0]))))
            summary_md += "\n"

        with open(f"output/{exp_name}/summary.md", "w") as f:
            f.write(summary_md)

        for _, row in tqdm.tqdm(list(fps.iterrows())):
            if "tensorflow" in row["file"]:
                logger.debug("Skipping over tf")

                continue
            commits = acquire_commits(
                row["file"], row["timestamp"][0], ignore_errors=True)
            if commits:
                with open(f'output/{exp_name}/{row["file"]}_{row["timestamp"][0].strftime("%Y-%m-%d")}.json',
                          'w+') as mfile:
                    json.dump(commits, mfile, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()
