#!/usr/bin/env python
# coding: utf-8
from importlib.metadata import metadata
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam

from models import *
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import KFold
from helper import find_best_f1, EnumAction, safe_mkdir
from classes import Repository
from matplotlib import pyplot as plt
from matplotlib import pyplot
from enum import Enum
from pandas import DataFrame
from keras_tuner import RandomSearch


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

import matplotlib
matplotlib.use('TkAgg')


logger = logging.getLogger(__name__)
coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s')


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
    for i in range(gap_days, cur_repo_data.shape[0]):
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
        res = cur_repo_data[event[1] - backs:event[1] + backs]

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


    # created at
    # fundingLinks
    # languages_edges


    
def create_dataset(aggr_options, benign_vuln_ratio, hours, days, resample, backs, metadata=False,comment=""):
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
        aggr_options, backs, benign_vuln_ratio, days, hours, resample,metadata,comment)
    safe_mkdir("ready_data")
    safe_mkdir("ready_data/" + dirname)

    counter = 0
    for file in (pbar := tqdm.tqdm(os.listdir(repo_dirs)[:])):
        tqdm_update = lambda cur: pbar.set_description(f"{file} - {cur}")

        if ".csv" not in file:
            continue
        file = file.split(".csv")[0]
        counter += 1

        repo_holder = Repository()
        repo_holder.file = file
        tqdm_update(f"read")
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

        if metadata:
            tqdm_update("add metadata")
            cur_repo = helper.add_metadata(cur_repo,file)


        tqdm_update("fix_repo_shape")
        cur_repo = fix_repo_shape(all_set, cur_repo,metadata=metadata, update = tqdm_update)

        vulns = cur_repo.index[cur_repo['Vuln'] == 1].tolist()
        if not len(vulns):
            continue
        benigns = cur_repo.index[cur_repo['Vuln'] == 0].tolist()
        random.shuffle(benigns)
        benigns = benigns[:benign_vuln_ratio*len(vulns)]

        cur_repo = cur_repo.drop(["Vuln"], axis=1)

        tqdm_update("extract_window")
        if aggr_options == Aggregate.none:
            cur_repo = add_time_one_hot_encoding(cur_repo, with_idx=True)
        elif aggr_options == Aggregate.before_cve:
            cur_repo = cur_repo.reset_index().drop(["created_at"], axis=1).set_index("idx")

        extract_window(aggr_options, hours, days, resample, backs,
                       file, repo_holder.vuln_lst, repo_holder.vuln_details, cur_repo, vulns, VULN_TAG)

        extract_window(aggr_options, hours, days, resample, backs,
                       file, repo_holder.benign_lst, repo_holder.benign_details, cur_repo, benigns, BENIGN_TAG)

        tqdm_update("pad")
        repo_holder.pad_repo()

        tqdm_update("save")
        with open("ready_data/" + dirname + "/" + repo_holder.file + ".pkl", 'wb') as f:
            pickle.dump(repo_holder, f)

        all_repos.append(repo_holder)

    with open("ready_data/"+dirname+"/column_names.pkl",'wb') as f:
        pickle.dump(cur_repo.columns,f)
    return all_repos, cur_repo.columns


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


def fix_repo_shape(all_set, cur_repo, metadata=False, update = lambda cur: None):
    """
    fixes the shape of the repo
    :param all_set: the set of all events
    :param cur_repo: the current repo
    :return: the fixed repo
    """
    cur_repo['created_at'] = pd.to_datetime(cur_repo['created_at'], utc=True)

    update("Removed Duplicates")
    cur_repo = cur_repo[~cur_repo.duplicated(
        subset=['created_at', 'Vuln'], keep='first')]

    update("Sorted and managed index")
    cur_repo = cur_repo.set_index(["created_at"])
    cur_repo = cur_repo.sort_index()
    cur_repo = cur_repo[cur_repo.index.notnull()]
    all_set.update(cur_repo.type.unique())
    cur_repo['idx'] = range(len(cur_repo))
    cur_repo = cur_repo.reset_index().set_index(["created_at", "idx"])

    update("Normalizing Data")
    integer_fields = ['Add', 'Del', 'Files']
    if metadata:
        integer_fields += ['diskUsage']

    for commit_change in integer_fields:
        if commit_change in cur_repo.columns:
            cur_repo[commit_change].fillna(0, inplace=True)
            cur_repo[commit_change] = cur_repo[commit_change].astype(int)
            cur_repo[commit_change] = (
                cur_repo[commit_change] - cur_repo[commit_change].mean()) / cur_repo[commit_change].std()

    update("One Hot encoding")
    cur_repo = add_type_one_hot_encoding(cur_repo)

    update("Droping unneeded columns")
    cur_repo = cur_repo.drop(["type"], axis=1)
    cur_repo = cur_repo.drop(["name"], axis=1)
    cur_repo = cur_repo.drop(["Unnamed: 0"], axis=1)
    cur_repo = cur_repo.drop(["Hash"], axis=1)
    return cur_repo


def make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days, hours, resample,metadata,comment):
    """
    :return: name of the directory to save the data in
    """
    comment = "_"+comment if comment else ""
    metadata = "_meta" if metadata else ""
    if aggr_options == Aggregate.before_cve:
        name_template = f"{str(aggr_options)}_R{benign_vuln_ratio}_B{backs}"+metadata+comment
    elif aggr_options == Aggregate.after_cve:
        name_template = f"{str(aggr_options)}_R{benign_vuln_ratio}_RE{resample}_H{hours}_D{days}"+metadata+comment
    elif aggr_options == Aggregate.none:
        name_template = f"{str(aggr_options)}_R{benign_vuln_ratio}_B{backs}"+metadata+comment
    else:
        raise Exception("Aggr options not supported")
    logger.debug(name_template)
    return name_template


def extract_dataset(aggr_options=Aggregate.none, benign_vuln_ratio=1, hours=0, days=10, resample=12, backs=50,
                    cache=False,metadata=False,comment = ""):
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
        aggr_options, backs, benign_vuln_ratio, days, hours, resample,metadata,comment)
    if cache and os.path.isdir("ready_data/" + dirname) and len(os.listdir("ready_data/" + dirname)) != 0:
        logger.info(f"Loading Dataset {dirname}")
        all_repos = []
        for file in os.listdir("ready_data/" + dirname):
            with open("ready_data/" + dirname + "/" + file, 'rb') as f:
                repo = pickle.load(f)
                all_repos.append(repo)
        column_names = pickle.load(open("ready_data/" + dirname + "/column_names.pkl", 'rb'))

    else:
        logger.info(f"Creating Dataset {dirname}")
        all_repos, column_names = create_dataset(
            aggr_options, benign_vuln_ratio, hours, days, resample, backs,metadata=metadata,comment=comment)

    return all_repos, dirname, column_names


def model_selector(model_name, shape1, shape2, optimizer):
    if model_name == "lstm":
        return lstm(shape1, shape2, optimizer)
    if model_name == "conv1d":
        return conv1d(shape1, shape2, optimizer)
    if model_name == "conv1dlstm":
        return conv1dlstm(shape1, shape2, optimizer)
    if model_name == "lstm_autoencoder":
        return lstm_autoencoder(shape1, shape2, optimizer)
    if model_name == "bilstm":
        return bilstm(shape1, shape2, optimizer)
    if model_name == "bigru":
        return bigru(shape1, shape2, optimizer)
    return getattr(models,model_name)(shape1,shape2,optimizer)



def feature_importance(model, X_train, columns):
    import shap
    regressor = model
    random_ind = np.random.choice(X_train.shape[0], 1000, replace=False)
    print(random_ind)
    data = X_train[random_ind[0:500]]
    e = shap.DeepExplainer((regressor.layers[0].input, regressor.layers[-1].output),data)
    test1 = X_train[random_ind[500:1000]]
    shap_val = e.shap_values(test1)
    shap_val = np.array(shap_val)
    shap_val = np.reshape(shap_val,(int(shap_val.shape[1]),int(shap_val.shape[2]),int(shap_val.shape[3])))
    shap_abs = np.absolute(shap_val)
    sum_0 = np.sum(shap_abs,axis=0)
    f_names = columns
    x_pos = [i for i, _ in enumerate(f_names)]
    plt1 = plt.subplot(311)
    plt1.barh(x_pos,sum_0[1])
    plt1.set_yticks(x_pos)
    plt1.set_yticklabels(f_names)
    plt1.set_title("yesterday features (time-step 2)")
    plt2 = plt.subplot(312,sharex=plt1)
    plt2.barh(x_pos,sum_0[0])
    plt2.set_yticks(x_pos)
    plt2.set_yticklabels(f_names)
    plt2.set_title("The day before yesterday’s features(time-step 1)")
    plt.tight_layout()
    plt.show()

    with open("tmpname",'wb') as mfile:
        np.save(mfile,f_names)
    with open("tmpsum",'wb') as mfile:
        np.save(mfile,sum_0)

def train_model(X_train, y_train, X_val, y_val, exp_name, batch_size=32,  epochs=20, model_name="LSTM", columns = []):
    """
    Evaluate the model with the given data.
    """

    optimizer = SGD(learning_rate=0.01, momentum=0.9,
                    nesterov=True, clipnorm=1.)
    optimizer = Adam(learning_rate=0.001)
    # Create the model
    model = model_selector(
        model_name, X_train.shape[1], X_train.shape[2], optimizer)
    safe_mkdir("models/")
    safe_mkdir("models/" + exp_name)

    best_model_path = f'models/{exp_name}/mdl_wts.hdf5'
    mcp_save = ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_accuracy', mode='max')

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=500)

    verbose = 0
    if logger.level < logging.CRITICAL:
        verbose = 1

    validation_data = (X_val, y_val) if len(X_val) else None
    history = model.fit(X_train, y_train, verbose=verbose, epochs=epochs, shuffle=True, batch_size=batch_size,
                        validation_data=validation_data, callbacks=[mcp_save])

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


def check_results(X_test, y_test, pred, model, exp_name, model_name, save=False):
    """
    Check the results of the model.
    """
    used_y_test = np.asarray(y_test).astype('float32')
    scores = model.evaluate(X_test, used_y_test, verbose=0)
    if len(scores) == 1:
        return 0
    max_f1, thresh, _ = find_best_f1(X_test, used_y_test, model)
    logger.critical(f"{max_f1}, {thresh},{str(scores[1])}")
    if save:
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
    parser.add_argument('--comment', action='store',
                        type=str, help='add comment to results.')
    parser.add_argument('--hypertune',action="store_true", help="Should hypertune parameter")
    parser.add_argument('--batch', type=int, default=64, help="Batch size")
    parser.add_argument('--metadata',action="store_true", help="Use metadata")
    args = parser.parse_args()
    return args


def split_into_x_and_y(repos, with_details=False, remove_unimportant_features=False):
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


    if remove_unimportant_features:
        important_features =np.load("important_features.npy")
        X_train = X_train[:,:, important_features]

    if with_details:
        if details:
            details = np.concatenate(details)
        return X_train, y_train, details

    return X_train, y_train

def split_repos_into_train_and_validation(X_train,y_train):
    raise NotImplementedError()

def hypertune(X_train,y_train,X_test,y_test):
    tuner = RandomSearch(hypertune_gru(X_train.shape[1], X_train.shape[2]),
                         objective='val_accuracy',
                         max_trials=10,
                         executions_per_trial=10,
                         directory='tuner1',
                         project_name='hypertune_gru')
                         
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=60)

    tuner.search(X_train,y_train,epochs=50, validation_data=(X_test, y_test),verbose=0,callbacks=[es])
    tuner.results_summary()
    print(tuner.get_best_hyperparameters(1))
    return tuner.get_best_models(1)[0]

def init():
    SEED = 0x1337
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)


def extract_fp(x_test, y_test, pred, test_details, exp_name,):
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


def split_repos(repos, train_size):  
    train_repos = []
    test_repos = []
    vuln_counter = 0
    train_repo_counter = 0
    for repo in repos:
        cur_vuln_counter = repo.get_num_of_vuln()
        if(vuln_counter + cur_vuln_counter < train_size):
            train_repo_counter += 1
            train_repos.append(repo)
        else:
            test_repos.append(repo)
        vuln_counter += cur_vuln_counter
    return train_repos, test_repos, train_repo_counter

def main():
    args = parse_args()
    if not args.loglevel:
        logger.level = logging.CRITICAL
    else:
        logger.level = args.loglevel

    init()

    all_repos, exp_name, columns = extract_dataset(aggr_options=args.aggr,
                                          resample=args.resample,
                                          benign_vuln_ratio=args.ratio,
                                          hours=args.hours,
                                          days=args.days,
                                          backs=args.backs,
                                          cache=args.cache,
                                          metadata = args.metadata,
                                          comment=args.comment)

    to_pad = 0
    num_of_vulns = 0

    random.shuffle(all_repos)
    all_repos = [repo for repo in all_repos if getattr(repo,"get_num_of_vuln",None) is not None]

    for idx,repo in enumerate(all_repos):
        num_of_vulns += repo.get_num_of_vuln()
        if len(repo.get_all_lst()[0].shape) > 1:
            to_pad = max(to_pad, repo.get_all_lst()[0].shape[1])
        else:
            all_repos.remove(repo)

    for repo in all_repos:
        repo.pad_repo(to_pad=to_pad)

    TRAIN_SIZE = 0.8
    VALIDATION_SIZE = 0.1
    train_size = int(TRAIN_SIZE * num_of_vulns)
    validation_size = int(VALIDATION_SIZE * num_of_vulns)
    test_size = num_of_vulns - train_size - validation_size

    logger.info(f"Train size: {train_size}")
    logger.info(f"Validation size: {validation_size}")
    logger.info(f"Test size: {test_size}")


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
    train_and_val_repos, test_repos, _ = split_repos(all_repos, train_size + validation_size)


    best_val_accuracy = 0
    idx = 0
    best_fold_x_train = None
    best_fold_y_train = None
    best_fold_x_val = None
    best_fold_y_val = None
    remove_unimportant = True

    for i in range(args.kfold):

        print(i, len(train_and_val_repos),train_size)
        if i ==7:
            print("H")
        train_repos, val_repos, num_of_train_repos = split_repos(train_and_val_repos, train_size)
        X_train,y_train = split_into_x_and_y(train_repos, remove_unimportant_features=remove_unimportant)
        X_val,y_val = split_into_x_and_y(val_repos, remove_unimportant_features=remove_unimportant)



        print(f"all_repos = {len(all_repos)}, train_repos = {len(train_repos)}, val_repos = {len(val_repos)}")
        print(f"x_train = {len(X_train)}, y_train = {len(y_train)}, x_val = {len(X_val)}, y_val = {len(y_val)}")
        print(f"train ratio = {len(y_train[y_train == 1]) / len(y_train[y_train == 0])}")
        print(f"val ratio = {len(y_val[y_val == 1]) / len(y_val[y_val == 0])}")
        print(f"train val  ratio = {len(X_train)/len(X_val)}")

        model = train_model(X_train, y_train, X_val, y_val,
                            exp_name, batch_size=args.batch, epochs=args.epochs, model_name=args.model, columns = columns)

        pred = model.predict(X_val, verbose = 0).reshape(-1)

        acc = check_results(X_val, y_val, pred, model, exp_name, args.model)
        
        if acc>best_val_accuracy:
            best_model = model
            best_val_accuracy = acc
            best_fold_x_train = X_train
            best_fold_y_train = y_train
            best_fold_x_val = X_val
            best_fold_y_val = y_val

        num_of_val_repos = len(train_and_val_repos)-num_of_train_repos
        train_and_val_repos = train_and_val_repos[-num_of_val_repos:] + train_and_val_repos[:-num_of_val_repos]


    logging.critical(f"Best val accuracy: {best_val_accuracy}")
    if args.hypertune:
        best_model = hypertune(best_fold_x_train,best_fold_y_train,best_fold_x_val, best_fold_y_val)
    
    # handle test set
    X_test,y_test,test_details = split_into_x_and_y(test_repos, with_details=True,remove_unimportant_features=remove_unimportant)
    pred = best_model.predict(X_test, verbose = 0).reshape(-1)
    acc = check_results(X_test, y_test, pred, best_model, exp_name, args.model,save=True)
    if args.fp:
            extract_fp(X_test,y_test,pred,test_details,exp_name)

if __name__ == '__main__':
    main()
