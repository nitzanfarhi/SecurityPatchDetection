#!/usr/bin/env python
# coding: utf-8
from importlib.metadata import metadata
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam


from data_collection.create_dataset import gh_cve_dir, repo_metadata_filename
from dataset_utils import Aggregate, extract_dataset
from helper import find_best_accuracy, find_best_f1, EnumAction, safe_mkdir
from helper import Repository
from models import *
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from matplotlib import pyplot
from pandas import DataFrame
from keras_tuner import RandomSearch, Hyperband
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
import models

import coloredlogs
import logging
import warnings

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

matplotlib.use('TkAgg')

logger = logging.getLogger(__name__)
coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s')


BENIGN_EVENTS_RETRY = 5


def find_benign_events(cur_repo_data, gap_days, num_of_events):
    """
    :param cur_repo_data: DataFrame that is processed
    :param gap_days: number of days to look back for the events
    :param num_of_events: number of events to find
    :return: list of all events
    """
    benign_events = []
    retries = num_of_events * BENIGN_EVENTS_RETRY
    counter = 0
    for _ in range(num_of_events):
        found_event = False
        while not found_event:
            if counter >= retries:
                return benign_events
            try:
                cur_event = random.randint(
                    2 * gap_days + 1,
                    cur_repo_data.shape[0] - gap_days * 2 - 1)
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
    for i in range(gap_days, cur_repo_data.shape[0]):
        event = cur_repo_data.index[i]
        before_vuln = event - gap_days
        res_event = cur_repo_data[before_vuln:event - 1]
        all_events.append(res_event.iloc[:, :-1].values)
        labels.append(res_event.iloc[:, -1].values)
    return all_events, labels






def add_time_one_hot_encoding(df, with_idx=False):
    """
    :param df: dataframe to add time one hot encoding to
    :param with_idx: if true, adds index column to the dataframe
    :return: dataframe with time one hot encoding
    """

    hour = pd.get_dummies(df.index.get_level_values(0).hour.astype(
        pd.CategoricalDtype(categories=range(24))),
                          prefix='hour')
    week = pd.get_dummies(df.index.get_level_values(0).dayofweek.astype(
        pd.CategoricalDtype(categories=range(7))),
                          prefix='day_of_week')
    day_of_month = pd.get_dummies(df.index.get_level_values(0).day.astype(
        pd.CategoricalDtype(categories=range(1, 32))),
                                  prefix='day_of_month')

    df = pd.concat([df.reset_index(), hour, week, day_of_month], axis=1)
    if with_idx:
        df = df.set_index(['created_at', 'idx'])
    else:
        df = df.set_index(['index'])
    return df



repo_dirs = 'data_collection/gh_cve_proccessed'
repo_metadata = 'data_collection/repo_metadata.json'
benign_all, vuln_all = [], []
n_features = 0
gap_days = 150







def model_selector(model_name, shape1, shape2, optimizer):
    return getattr(models, model_name)(shape1, shape2, optimizer)


def feature_importance(model, X_train, columns):
    import shap
    regressor = model
    random_ind = np.random.choice(X_train.shape[0], 1000, replace=False)
    data = X_train[random_ind[:500]]
    e = shap.DeepExplainer(
        (regressor.layers[0].input, regressor.layers[-1].output), data)
    test1 = X_train[random_ind[500:1000]]
    shap_val = e.shap_values(test1)
    shap_val = np.array(shap_val)
    shap_val = np.reshape(shap_val, (int(
        shap_val.shape[1]), int(shap_val.shape[2]), int(shap_val.shape[3])))
    shap_abs = np.absolute(shap_val)
    sum_0 = np.sum(shap_abs, axis=0)
    f_names = columns
    x_pos = [i for i, _ in enumerate(f_names)]
    plt1 = plt.subplot(311)
    plt1.barh(x_pos, sum_0[1])
    plt1.set_yticks(x_pos)
    plt1.set_yticklabels(f_names)
    plt1.set_title("yesterday features (time-step 2)")
    plt2 = plt.subplot(312, sharex=plt1)
    plt2.barh(x_pos, sum_0[0])
    plt2.set_yticks(x_pos)
    plt2.set_yticklabels(f_names)
    plt2.set_title("The day before yesterday’s features(time-step 1)")
    plt.tight_layout()
    plt.show()

    with open("tmpname", 'wb') as mfile:
        np.save(mfile, f_names)
    with open("tmpsum", 'wb') as mfile:
        np.save(mfile, sum_0)


def train_model(X_train,
                y_train,
                X_val,
                y_val,
                exp_name,
                batch_size=32,
                epochs=20,
                model_name="LSTM",
                columns=None):
    """
    Evaluate the model with the given data.
    """

    if columns is None:
        columns = []
    optimizer = SGD(learning_rate=0.01,
                    momentum=0.9,
                    nesterov=True,
                    clipnorm=1.)
    optimizer = Adam(learning_rate=0.001)
    # Create the model
    model = model_selector(model_name, X_train.shape[1], X_train.shape[2],
                           optimizer)
    safe_mkdir("models/")
    safe_mkdir("models/" + exp_name)

    best_model_path = f'models/{exp_name}/mdl_wts.hdf5'
    mcp_save = ModelCheckpoint(best_model_path,
                               save_best_only=True,
                               monitor='val_accuracy',
                               mode='max')

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=500)

    verbose = 1 if logger.level < logging.CRITICAL else 0
    validation_data = (X_val, y_val) if len(X_val) else None
    history = model.fit(X_train,
                        y_train,
                        verbose=verbose,
                        epochs=epochs,
                        shuffle=True,
                        batch_size=batch_size,
                        validation_data=validation_data,
                        callbacks=[mcp_save])

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
        f"figs/{exp_name}_{epochs}_{model_name}_t{last_training}_v{last_validation}.png"
    )
    # pyplot.show()

    # Final evaluation of the model
    return tf.keras.models.load_model(best_model_path)


def acquire_commits(name, date, ignore_errors=False):
    """
    Acquire the commits for the given repository.
    """
    group, repo = name.replace(".csv", "").split("_", 1)

    github_format = "%Y-%m-%dT00:00:00"
    for branch in ["master", "main"]:
        res = helper.run_query(helper.commits_between_dates.format(
            group, repo, branch, date.strftime(github_format),
            (date + datetime.timedelta(days=1)).strftime(github_format)),
                               ignore_errors=ignore_errors)
        if "data" in res and "repository" in res["data"] and "object" in res[
                'data']['repository']:
            obj = res['data']['repository']['object']
            if obj is None:
                continue
            if "history" in obj:
                return res['data']['repository']['object']['history']['nodes']
    return ""


def check_results(X_test,
                  y_test,
                  pred,
                  model,
                  exp_name,
                  model_name,
                  save=False):
    """
    Check the results of the model.
    """
    used_y_test = np.asarray(y_test).astype('float32')
    scores = model.evaluate(X_test, used_y_test, verbose=0)
    if len(scores) == 1:
        return 0
    max_f1, f1_thresh, _ = find_best_f1(X_test, used_y_test, model)
    max_acc, acc_thresh, _ = find_best_accuracy(X_test, used_y_test, model)
    logger.critical(f"F1 - {max_f1}, {f1_thresh}")
    logger.critical(f"Acc - {max_acc}, {acc_thresh}")
    if save:
        with open(f"results/{exp_name}_{model_name}.txt", 'w') as mfile:
            mfile.write('Accuracy: %.2f%%\n' % (max_acc * 100))
            mfile.write('fscore: %.2f%%\n' % (max_f1 * 100))
            mfile.write('confusion matrix:\n')
            tn, fp, fn, tp = confusion_matrix(y_test, pred > acc_thresh).ravel()
            conf_matrix = f"tn={tn}, fp={fp}, fn={fn}, tp={tp}"
            mfile.write(conf_matrix)

            logger.critical('Accuracy: %.2f%%' % (max_acc * 100))
            logger.critical('fscore: %.2f%%' % (max_f1 * 100))
            logger.critical(str(conf_matrix))

        fpr = {}
        tpr = {}
        fpr["micro"], tpr["micro"], _ = roc_curve(used_y_test, pred)
        roc_auc = {"micro": auc(fpr["micro"], tpr["micro"])}
        plt.figure()
        lw = 2

        plt.plot(fpr['micro'],
                 tpr['micro'],
                 color="darkorange",
                 lw=lw,
                 label="ROC curve (area = %0.2f)" % roc_auc['micro'])
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.legend(loc="lower right")
        plt.savefig(f"figs/auc_{exp_name}_{roc_auc['micro']}.png")

    return max_acc


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hours', type=int, default=0, help='hours back')
    parser.add_argument('-d', '--days', type=int, default=10, help='days back')
    parser.add_argument('--resample',
                        type=int,
                        default=24,
                        help='number of hours that should resample aggregate')
    parser.add_argument('-r',
                        '--ratio',
                        type=int,
                        default=1,
                        help='benign vuln ratio')
    parser.add_argument('-a',
                        '--aggr',
                        type=Aggregate,
                        action=EnumAction,
                        default=Aggregate.none)
    parser.add_argument('-b',
                        '--backs',
                        type=int,
                        default=10,
                        help=' using none aggregation, operations back')
    parser.add_argument('-v',
                        '--verbose',
                        help="Be verbose",
                        action="store_const",
                        dest="loglevel",
                        const=logging.DEBUG)
    parser.add_argument('-c',
                        '--cache',
                        '--cached',
                        help="Use Cached Data",
                        action="store_const",
                        dest="cache",
                        const=True)
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=10,
                        help=' using none aggregation, operations back')
    parser.add_argument('-f',
                        '--find-fp',
                        help="Find False positive commits",
                        action="store_const",
                        dest="fp",
                        const=True)
    parser.add_argument('-m',
                        '--model',
                        action='store',
                        type=str,
                        help='The model to receive.')
    parser.add_argument('-k',
                        '--kfold',
                        type=int,
                        default=10,
                        help="Kfold cross validation")
    parser.add_argument('--comment',
                        action='store',
                        type=str,
                        help='add comment to results.')
    parser.add_argument('--hypertune',
                        action="store_true",
                        help="Should hypertune parameter")
    parser.add_argument('--batch', type=int, default=64, help="Batch size")
    parser.add_argument('--metadata', action="store_true", help="Use metadata")
    parser.add_argument('--submodels',
                        action="store_true",
                        help="Use metadata")
    parser.add_argument('--data-location', action="store", help="Data location", default=r"data_collection\data")
    parser.add_argument(
        '--merge-all',
        action="store_true",
        help="Merge all repositories before splitting into sets")

    return parser.parse_args()


def split_into_x_and_y(repos,
                       with_details=False,
                       remove_unimportant_features=False):
    """
    Split the repos into X and Y.
    """
    if len(repos) == 0:
        raise ValueError("No repos to split")

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

    # if remove_unimportant_features:
    #     important_features = np.load("important_features.npy")
    #     X_train = X_train[:, :, important_features]

    if with_details:
        if details:
            details = np.concatenate(details)
        return X_train, y_train, details

    return X_train, y_train


def split_repos_into_train_and_validation(X_train, y_train):
    raise NotImplementedError()

def hypertune(X_train,y_train,X_test,y_test):
    tuner = Hyperband(hypertune_gru_cnn(X_train.shape[1], X_train.shape[2]),
                         objective='val_accuracy',
                         # max_trials=10,
                         executions_per_trial=10,
                         directory='hypertune',
                         project_name='hyper_gru_2')
                         
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=60)
    while True:
        try:
            tuner.search(X_train,
                y_train,
                batch_size=64,
                epochs=500,
                validation_data=(X_test, y_test),
                verbose=1,
                callbacks=[es])
            break
        except Exception as e:
            print(e)
            continue
        
    tuner.results_summary()
    print(tuner.get_best_hyperparameters(1))
    return tuner.get_best_models(1)[0]


def init():
    SEED = 0x0
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)


def analyze_repos(y_test, pred, details):
    df = DataFrame(zip(y_test, pred, details[:, 0], details[:, 1]),
                   columns=['real', 'pred', 'file', 'timestamp'])

    m_name, m_group, m_fps, m_fns = [], [], [], []
    for name, group in df.groupby('file'):
        group_fps = group[(group['pred'] > group['real'])
                          & (group['pred'] > 0.5)]
        group_fns = group[(group['pred'] < group['real'])
                          & (group['pred'] <= 0.5)]
        m_name.append(name)
        m_group.append(len(group))
        m_fps.append(len(group_fps))
        m_fns.append(len(group_fns))
    return pd.DataFrame({
        "name": m_name,
        "group": m_group,
        "fps": m_fps,
        "fns": m_fns
    })


def extract_fp(
    x_test,
    y_test,
    pred,
    test_details,
    exp_name,
):
    safe_mkdir("output")
    safe_mkdir(f"output/{exp_name}")

    summary_md = ""
    summary_md += f"# {exp_name}\n"

    df = DataFrame(zip(y_test, pred, test_details[:, 0], test_details[:, 1]),
                   columns=['real', 'pred', 'file', 'timestamp'])

    fps = df[(df['pred'] > df['real']) & (df['pred'] > 0.5)]

    groups = fps.groupby('file')
    for name, group in groups:
        summary_md += f"## {name}\n"
        summary_md += "\n".join(
            list(group['timestamp'].apply(lambda x: f"* {str(x[0])}")))
        summary_md += "\n"

    with open(f"output/{exp_name}/summary.md", "w") as f:
        f.write(summary_md)

    for _, row in tqdm.tqdm(list(fps.iterrows())):
        if "tensorflow" in row["file"]:
            logger.debug("Skipping over tf")

            continue
        if commits := acquire_commits(row["file"],
                                      row["timestamp"][0],
                                      ignore_errors=True):
            with open(
                    f'output/{exp_name}/{row["file"]}_{row["timestamp"][0].strftime("%Y-%m-%d")}.json',
                    'w+') as mfile:
                json.dump(commits, mfile, indent=4, sort_keys=True)


def split_repos(repos, train_size):
    train_repos = []
    test_repos = []
    vuln_counter = 0
    train_repo_counter = 0
    for repo in repos:
        cur_vuln_counter = repo.get_num_of_vuln()
        if (vuln_counter + cur_vuln_counter < train_size):
            train_repo_counter += 1
            train_repos.append(repo)
        else:
            test_repos.append(repo)
        vuln_counter += cur_vuln_counter
    return train_repos, test_repos, train_repo_counter


def test_submodels(all_repos, exp_name, columns, args):
    logging.critical("--- Checking Boolean Variables ---")
    test_bool(all_repos, exp_name, columns, args)
    logging.critical("--- Checking Hour Ranges ---")
    test_hour_ranges(all_repos, exp_name, columns, args)
    logging.critical("--- Checking Programming Languages ---")
    test_languages(all_repos, exp_name, columns, args)


def test_repos(repos, exp_name, columns, args, train_size=0.7):
    if not repos:
        return 0
    train_size = sum(repo.get_num_of_vuln() for repo in repos) * train_size
    train_repos, val_repos, _ = split_repos(repos, train_size)
    if not train_repos or not val_repos:
        return 0
    X_train, y_train = split_into_x_and_y(train_repos,
                                          remove_unimportant_features=True)
    X_val, y_val = split_into_x_and_y(val_repos,
                                      remove_unimportant_features=True)

    model = train_model(X_train,
                        y_train,
                        X_val,
                        y_val,
                        exp_name,
                        batch_size=args.batch,
                        epochs=args.epochs,
                        model_name=args.model,
                        columns=columns)

    pred = model.predict(X_val, verbose=0).reshape(-1)
    return check_results(X_val, y_val, pred, model, exp_name, args.model)


def test_bool(all_repos, exp_name, columns, args):
    for bool in bool_metadata:
        falses, trues = [], []
        for repo in all_repos:
            if bool in repo.metadata and repo.metadata[bool]:
                trues.append(repo)
            else:
                falses.append(repo)
        true_acc = test_repos(trues, exp_name, columns, args)
        false_acc = test_repos(falses, exp_name, columns, args)
        logging.critical(f"--- {bool} --- True: {true_acc} False: {false_acc}")


def test_languages(all_repos, exp_name, columns, args):
    languages = ["PHP", "HTML", "JavaScript", "C", "C++", "Perl", "Python"]
    for language in languages:
        repos = [
            repo for repo in all_repos
            if language in repo.metadata["languages_edges"]
        ]
        train_size = sum(repo.get_num_of_vuln() for repo in repos) * 0.7
        train_repos, val_repos, _ = split_repos(repos, train_size)
        X_train, y_train = split_into_x_and_y(train_repos,
                                              remove_unimportant_features=True)
        X_val, y_val = split_into_x_and_y(val_repos,
                                          remove_unimportant_features=True)

        model = train_model(X_train,
                            y_train,
                            X_val,
                            y_val,
                            exp_name,
                            batch_size=args.batch,
                            epochs=args.epochs,
                            model_name=args.model,
                            columns=columns)

        pred = model.predict(X_val, verbose=0).reshape(-1)

        acc = check_results(X_val, y_val, pred, model, exp_name, args.model)
        print(language)
        print(acc)


def test_hour_ranges(all_repos, exp_name, columns, args):
    hour_ranges = [
        (-12, -7), (-6, -1), (0, 0), (1, 1), 
        (2, 2,), (3, 7), (8, 14)]

    hour_repo_array = [[] for _ in range(len(hour_ranges))]
    for repo in all_repos:
        timezone = repo.metadata["timezone"]
        for rng in hour_ranges:
            if timezone >= rng[0] and timezone <= rng[1]:
                hour_repo_array[hour_ranges.index(rng)].append(repo)
                break
    for idx, repos in enumerate(hour_repo_array):
        train_size = sum(repo.get_num_of_vuln() for repo in repos) * 0.7
        train_repos, val_repos, _ = split_repos(repos, train_size)
        X_train, y_train = split_into_x_and_y(train_repos,
                                              remove_unimportant_features=True)
        X_val, y_val = split_into_x_and_y(val_repos,
                                          remove_unimportant_features=True)

        model = train_model(X_train,
                            y_train,
                            X_val,
                            y_val,
                            exp_name,
                            batch_size=args.batch,
                            epochs=args.epochs,
                            model_name=args.model,
                            columns=columns)

        pred = model.predict(X_val, verbose=0).reshape(-1)

        acc = check_results(X_val, y_val, pred, model, exp_name, args.model)
        print(hour_ranges[idx])
        print(acc)


def main():
    args = parse_args()
    logger.level = args.loglevel or logging.CRITICAL
    init()
    all_repos, exp_name, columns = extract_dataset(
        aggr_options=args.aggr,
        resample=args.resample,
        benign_vuln_ratio=args.ratio,
        hours=args.hours,
        days=args.days,
        backs=args.backs,
        cache=args.cache,
        metadata=args.metadata,
        comment=args.comment,
        data_location=args.data_location)

    all_repos, num_of_vulns = pad_and_fix(all_repos)

    if args.submodels:
        return test_submodels(all_repos, exp_name, columns, args)

    TRAIN_SIZE = 0.8
    VALIDATION_SIZE = 0.1
    train_size = int(TRAIN_SIZE * num_of_vulns)
    validation_size = int(VALIDATION_SIZE * num_of_vulns)
    test_size = num_of_vulns - train_size - validation_size

    logger.info(f"Train size: {train_size}")
    logger.info(f"Validation size: {validation_size}")
    logger.info(f"Test size: {test_size}")

    if args.merge_all:
        best_val_accuracy = 0
        x_all, y_all = split_into_x_and_y(all_repos,
                                          remove_unimportant_features=False)
        X_train, X_test, y_train, y_test = train_test_split(x_all,
                                                            y_all,
                                                            test_size=0.2,
                                                            random_state=42)
        kf = KFold(n_splits=args.kfold, random_state=42, shuffle=True)
        for train_index, test_index in kf.split(X_train):
            cur_X_train, X_val = X_train[train_index], X_train[test_index]
            cur_y_train, y_val = y_train[train_index], y_train[test_index]
            model = train_model(cur_X_train,
                                cur_y_train,
                                X_val,
                                y_val,
                                exp_name,
                                batch_size=args.batch,
                                epochs=args.epochs,
                                model_name=args.model,
                                columns=columns)

            pred = model.predict(X_test, verbose=0).reshape(-1)
            acc = check_results(X_test, y_test, pred, model, exp_name,
                                args.model)
            if acc > best_val_accuracy:
                best_model = model
                best_val_accuracy = acc
                best_fold_x_train = cur_X_train
                best_fold_y_train = cur_y_train
                best_fold_x_val = X_val
                best_fold_y_val = y_val

        # handle test set
        pred = best_model.predict(X_test, verbose=0).reshape(-1)
        acc = check_results(X_test,
                            y_test,
                            pred,
                            best_model,
                            exp_name,
                            args.model,
                            save=True)
        logging.critical(f"Best test accuracy: {acc}")
        return
    # 1. choose k fold or hypertune
    # 2. Take 15% to test
    # 2. hypertune
    #   2.1 create train/val
    #   2.2 hypertune - run hypertune
    #   2.2 return model for result evaluation (4)
    # 3. Kfold
    #   3.1 create train/val
    #   3.2 create model
    #   3.3 run model and receive validation accuracy
    #   3.4 goto 3.2 k times, than select model with best accuracy and go to results
    # 4. results
    #   4.1 run model on test
    #   4.2 store accuracy and if fp is on run fp

    best_model = None
    train_and_val_repos, test_repos, _ = split_repos(
        all_repos, train_size + validation_size)

    best_val_accuracy = 0
    idx = 0
    best_fold_x_train = None
    best_fold_y_train = None
    best_fold_x_val = None
    best_fold_y_val = None
    remove_unimportant = True

    train_repos, val_repos, num_of_train_repos = split_repos(
        train_and_val_repos, train_size)
    X_train, y_train = split_into_x_and_y(
        train_repos, remove_unimportant_features=remove_unimportant)
    X_val, y_val = split_into_x_and_y(
        val_repos, remove_unimportant_features=remove_unimportant)

    for _ in range(args.kfold):
        train_repos, val_repos, num_of_train_repos = split_repos(
            train_and_val_repos, train_size)
        X_train, y_train = split_into_x_and_y(
            train_repos, remove_unimportant_features=remove_unimportant)
        X_val, y_val = split_into_x_and_y(
            val_repos, remove_unimportant_features=remove_unimportant)

        model = train_model(X_train,
                            y_train,
                            X_val,
                            y_val,
                            exp_name,
                            batch_size=args.batch,
                            epochs=args.epochs,
                            model_name=args.model,
                            columns=columns)

        pred = model.predict(X_val, verbose=0).reshape(-1)

        acc = check_results(X_val, y_val, pred, model, exp_name, args.model)

        if acc > best_val_accuracy:
            best_model = model
            best_val_accuracy = acc
            best_fold_x_train = X_train
            best_fold_y_train = y_train
            best_fold_x_val = X_val
            best_fold_y_val = y_val

        num_of_val_repos = len(train_and_val_repos) - num_of_train_repos
        train_and_val_repos = train_and_val_repos[-num_of_val_repos:] + \
            train_and_val_repos[:-num_of_val_repos]

    X_test, y_test, test_details = split_into_x_and_y(
        test_repos,
        with_details=True,
        remove_unimportant_features=remove_unimportant)

    logging.critical(f"Best val accuracy: {best_val_accuracy}")
    if args.hypertune:
        best_model = hypertune(X_train, y_train, X_test, y_test,
                               exp_name + f"_{args.model}")

    # handle test set
    X_test,y_test,test_details = split_into_x_and_y(test_repos, with_details=True,remove_unimportant_features=remove_unimportant)
    pred = best_model.predict(X_test, verbose = 0).reshape(-1)
    acc = check_results(X_test, y_test, pred, best_model, exp_name, args.model,save=True)
    logging.critical(f"Best test accuracy: {acc}")

    if args.fp:
        extract_fp(X_test, y_test, pred, test_details, exp_name)





if __name__ == '__main__':
    main()
