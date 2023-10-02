from enum import Enum
import logging
import os
import json
import numpy as np
import tqdm
import pandas as pd
import random 
import pickle 
from .helper import safe_mkdir
from .data_collection.create_dataset import gh_cve_dir, repo_metadata_filename
from .helper import find_best_accuracy, find_best_f1, EnumAction, safe_mkdir
from .helper import Repository, add_metadata
from . import helper, classes
import sys
sys.modules['helper'] = helper
sys.modules['classes'] = classes

DATASET_DIRNAME = "ready_data/"
event_types = [
    'PullRequestEvent', 'PushEvent', 'ReleaseEvent', 'DeleteEvent', 'issues',
    'CreateEvent', 'releases', 'IssuesEvent', 'ForkEvent', 'WatchEvent',
    'PullRequestReviewCommentEvent', 'stargazers', 'pullRequests', 'commits',
    'CommitCommentEvent', 'MemberEvent', 'GollumEvent', 'IssueCommentEvent',
    'forks', 'PullRequestReviewEvent', 'PublicEvent'
]


BENIGN_TAG = 0
VULN_TAG = 1

logger = logging.getLogger(__name__)

class Aggregate(Enum):
    none = "none"
    before_cve = "before"
    after_cve = "after"
    only_before = "only_before"


def make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days, hours,
                      resample, metadata, comment):
    """
    :return: name of the directory to save the data in
    """
    comment = f"_{comment}" if comment else ""
    metadata = "_meta" if metadata else ""
    if aggr_options in [Aggregate.before_cve, Aggregate.only_before, Aggregate.none]:
        name_template = f"{str(aggr_options)}_R{benign_vuln_ratio}_B{backs}{metadata}{comment}"
    elif aggr_options == Aggregate.after_cve:
        name_template = f"{str(aggr_options)}_R{benign_vuln_ratio}_RE{resample}_H{hours}_D{days}{metadata}{comment}"
    else:
        raise ValueError("Aggr options not supported")

    logger.debug(name_template)
    return name_template




def extract_window(aggr_options, hours, days, resample, backs, file,
                   window_lst, details_lst, cur_repo, cur_list, tag):
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
        res = get_event_window(cur_repo,
                               cur,
                               aggr_options,
                               days=days,
                               hours=hours,
                               backs=backs,
                               resample=resample)
        details = (file, cur, tag)
        window_lst.append(res)
        details_lst.append(details)


def create_dataset(data_path,
                   aggr_options,
                   benign_vuln_ratio,
                   hours,
                   days,
                   resample,
                   backs,
                   metadata=False,
                   comment="",
                   ):
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
    dirname = make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days,
                                hours, resample, metadata, comment)
    safe_mkdir(DATASET_DIRNAME)
    safe_mkdir(DATASET_DIRNAME + dirname)
    cve_dir = os.path.join(data_path, gh_cve_dir)
    all_metadata = json.load(open(os.path.join(data_path,repo_metadata_filename), 'r'))
    counter = 0
    for file in (pbar := tqdm.tqdm(os.listdir(cve_dir)[:])):

        def tqdm_update(cur):
            return pbar.set_description(f"{file} - {cur}")

        if ".csv" not in file:
            continue
        file = file.split(".csv")[0]
        counter += 1

        repo_holder = Repository()
        repo_holder.file = file
        tqdm_update("read")
        if not os.path.exists(os.path.join(cve_dir,file + ".parquet")):
            try:
                cur_repo = pd.read_csv(os.path.join(cve_dir,file + ".csv"),
                                       low_memory=False,
                                       parse_dates=['created_at'],
                                       dtype={
                                           "type": "string",
                                           "name": "string",
                                           "Hash": "string",
                                           "Add": np.float64,
                                           "Del": np.float64,
                                           "Files": np.float64,
                                           "Vuln": np.float64
                                       })
            except pd.errors.EmptyDataError:
                continue

            cur_repo.to_parquet(os.path.join(cve_dir, file + ".parquet"))

        else:
            cur_repo = pd.read_parquet(os.path.join(cve_dir, file + ".parquet"))

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
            cur_repo = add_metadata(data_path, all_metadata, cur_repo, file, repo_holder)

        tqdm_update("fix_repo_shape")
        cur_repo = fix_repo_shape(all_set,
                                  cur_repo,
                                  metadata=metadata,
                                  update=tqdm_update)

        vulns = cur_repo.index[cur_repo['Vuln'] == 1].tolist()
        if not len(vulns):
            continue
        benigns = cur_repo.index[cur_repo['Vuln'] == 0].tolist()
        random.shuffle(benigns)
        benigns = benigns[:benign_vuln_ratio * len(vulns)]

        cur_repo = cur_repo.drop(["Vuln"], axis=1)

        tqdm_update("extract_window")
        if aggr_options == Aggregate.none:
            cur_repo = add_time_one_hot_encoding(cur_repo, with_idx=True)
        elif aggr_options == Aggregate.before_cve:
            cur_repo = cur_repo.reset_index().drop(["created_at"],
                                                   axis=1).set_index("idx")

        extract_window(aggr_options, hours, days, resample, backs, file,
                       repo_holder.vuln_lst, repo_holder.vuln_details,
                       cur_repo, vulns, VULN_TAG)

        extract_window(aggr_options, hours, days, resample, backs, file,
                       repo_holder.benign_lst, repo_holder.benign_details,
                       cur_repo, benigns, BENIGN_TAG)

        tqdm_update("pad")
        repo_holder.pad_repo()

        tqdm_update("save")
        with open(DATASET_DIRNAME + dirname + "/" + repo_holder.file + ".pkl",
                  'wb') as f:
            pickle.dump(repo_holder, f)

        all_repos.append(repo_holder)

    assert all_repos
    with open(DATASET_DIRNAME + dirname + "/column_names.pkl", 'wb') as f:
        pickle.dump(cur_repo.columns, f)
    return all_repos, cur_repo.columns



def extract_dataset(aggr_options=Aggregate.none,
                    benign_vuln_ratio=1,
                    hours=0,
                    days=10,
                    resample=12,
                    backs=50,
                    cache=False,
                    metadata=False,
                    comment="",
                    data_location=r"data_collection\data",
                    cache_location=DATASET_DIRNAME):
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

    dirname = make_new_dir_name(aggr_options, backs, benign_vuln_ratio, days,
                                hours, resample, metadata, comment)
    path_name = os.path.join(cache_location,dirname)
    if (cache and os.path.isdir(path_name)
            and len(os.listdir(path_name)) != 0
            and os.path.isfile(os.path.join(path_name,"column_names.pkl"))):

        logger.info(f"Loading Dataset {dirname}")
        all_repos = []
        try:
            for file in os.listdir(path_name):
                with open(os.path.join(path_name, file), 'rb') as f:
                    repo = pickle.load(f)
                    all_repos.append(repo)
            column_names = pickle.load(open(os.path.join(path_name, "column_names.pkl"), 'rb'))
        except AttributeError:
            logger.info(f"Malformed dataset - Creating Dataset {dirname}")
            all_repos, column_names = create_dataset(data_location, 
                aggr_options, benign_vuln_ratio, hours, days, resample, backs,metadata=metadata,comment=comment)            
    else:
        logger.info(f"Creating Dataset {dirname}")
        all_repos, column_names = create_dataset(data_location,
                                                 aggr_options,
                                                 benign_vuln_ratio,
                                                 hours,
                                                 days,
                                                 resample,
                                                 backs,
                                                 metadata=metadata,
                                                 comment=comment)

    return all_repos, dirname, column_names


def fix_repo_shape(all_set, cur_repo, metadata=False, update=lambda cur: None):
    """
    fixes the shape of the repo
    :param all_set: the set of all events
    :param cur_repo: the current repo
    :return: the fixed repo
    """
    cur_repo['created_at'] = pd.to_datetime(cur_repo['created_at'], utc=True)

    update("Removed Duplicates")
    cur_repo = cur_repo[
        ~cur_repo.duplicated(subset=['created_at', 'Vuln'], keep='first')]

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
            cur_repo[commit_change] = (cur_repo[commit_change] -
                                       cur_repo[commit_change].mean()
                                       ) / cur_repo[commit_change].std()

    update("One Hot encoding")
    cur_repo = add_type_one_hot_encoding(cur_repo)

    update("Droping unneeded columns")
    cur_repo = cur_repo.drop(["type"], axis=1)
    cur_repo = cur_repo.drop(["name"], axis=1)
    cur_repo = cur_repo.drop(["Unnamed: 0"], axis=1)
    cur_repo = cur_repo.drop(["Hash"], axis=1)
    return cur_repo


def add_type_one_hot_encoding(df):
    """
    :param df: dataframe to add type one hot encoding to
    :return: dataframe with type one hot encoding
    """
    type_one_hot = pd.get_dummies(
        df.type.astype(pd.CategoricalDtype(categories=event_types)))
    df = pd.concat([df, type_one_hot], axis=1)
    return df



def get_event_window(cur_repo_data,
                     event,
                     aggr_options,
                     days=10,
                     hours=10,
                     backs=50,
                     resample=24):
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
    if aggr_options == Aggregate.after_cve:
        cur_repo_data = cur_repo_data.reset_index().drop(
            ["idx"], axis=1).set_index("created_at")
        cur_repo_data = cur_repo_data.sort_index()
        hours_befs = 2

        indicator = event[0] - datetime.timedelta(days=0, hours=hours_befs)
        starting_time = indicator - datetime.timedelta(days=days, hours=hours)
        res = cur_repo_data[starting_time:indicator]
        new_row = pd.DataFrame([[0] * len(res.columns)],
                               columns=res.columns,
                               index=[starting_time])
        res = pd.concat([new_row, res], ignore_index=False)
        res = res.resample(f'{resample}H').sum()
        res = add_time_one_hot_encoding(res, with_idx=False)

    elif aggr_options == Aggregate.before_cve:
        res = cur_repo_data[event[1] - backs:event[1] + backs]

    elif aggr_options == Aggregate.only_before:
        res = cur_repo_data[event[1] - backs -1 :event[1]-1]

    elif aggr_options == Aggregate.none:
        res = cur_repo_data.reset_index().drop(
            ["created_at"],
            axis=1).set_index("idx")[event[1] - backs:event[1] + befs]
    return res.values

def pad_and_fix(all_repos):
    to_pad = 0
    num_of_vulns = 0
    random.shuffle(all_repos)
    all_repos = [
        repo for repo in all_repos
        if getattr(repo, "get_num_of_vuln", None) is not None
    ]

    for repo in all_repos:
        num_of_vulns += repo.get_num_of_vuln()
        if len(repo.get_all_lst()[0].shape) > 1:
            to_pad = max(to_pad, repo.get_all_lst()[0].shape[1])
        else:
            all_repos.remove(repo)

    for repo in all_repos:
        repo.pad_repo(to_pad=to_pad)
    return all_repos, num_of_vulns


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