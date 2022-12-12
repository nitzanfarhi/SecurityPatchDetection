import argparse
import contextlib
import json
import os
import subprocess
import traceback
import logging

import pandas as pd
import wget
import graphql

from pathlib import Path
from urllib.parse import urlparse
from git2json import *
from datetime import datetime

logging.basicConfig(
    filename='last_run.log',
    filemode='w',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG)

logger = logging.getLogger('analyze_cve')
logger.setLevel(logging.DEBUG)

logger.addHandler(logging.StreamHandler())

GITHUB_ARCHIVE_DIRNAME = "gharchive"
gh_cve_dir = "gh_cve_proccessed"
commit_directory = "commits"
json_commit_directory = "json_commits"
timezone_directory = "timezones"

LOG_GRAPHQL = "graphql_errlist.txt"
LOG_AGGR_ALL = 'gharchive_errlist.txt'
key_list = set()
err_counter = 0
ref_keys = [
    'ASCEND', 'ENGARDE', 'VIM', 'ERS', 'ATSTAKE', 'JVNDB', 'SLACKWARE',
    'ENGARDE', 'OPENBSD', 'CIAC', 'IBM', 'SUNALERT', 'FARMERVENEMA', 'XF',
    'ALLAIRE', 'VULN-DEV', 'MSKB', 'VULNWATCH', 'AIXAPAR', 'CERT-VN',
    'NTBUGTRAQ', 'XF', 'SUSE', 'CONECTIVA', 'SEKURE', 'MISC', 'MSKB', 'SUNBUG',
    'TURBO', 'VUPEN', 'BUGTRAQ', 'BUGTRAQ', 'DEBIAN', 'SCO', 'MS', 'IDEFENSE',
    'MLIST', 'INFOWAR', 'SECUNIA', 'FULLDISC', 'SUN', 'KSRT', 'HP', 'BID',
    'EL8', 'MANDRAKE', 'IMMUNIX', 'SECTRACK', 'VULN-DEV', 'CONFIRM', 'GENTOO',
    'SECTRACK', 'EL8', 'VUPEN', 'CERT', 'FREEBSD', 'HERT', 'L0PHT', 'BID',
    'CONECTIVA', 'SREASONRES', 'SCO', 'FEDORA', 'NAI', 'AUSCERT', 'ISS',
    'COMPAQ', 'NETECT', 'SUNBUG', 'CHECKPOINT', ' 1.23.1)', 'WIN2KSEC', 'BEA',
    'EXPLOIT-DB', 'KSRT', 'MANDRAKE', 'FRSIRT', 'JVN', 'RSI', 'NETBSD',
    'AUSCERT', 'OPENPKG', 'OPENBSD', 'CERT-VN', 'SUN', 'SECUNIA', 'VIM',
    'GENTOO', 'REDHAT', 'MS', 'COMPAQ', 'OVAL', 'CALDERA', 'FEDORA', 'FREEBSD',
    'CISCO', 'CISCO', 'WIN2KSEC', 'MANDRIVA', 'OSVDB', 'UBUNTU', 'EEYE', 'BEA',
    'IDEFENSE', 'NETBSD', 'SGI', 'SREASON', 'OSVDB', 'CIAC', 'BINDVIEW',
    'FULLDISC', 'NTBUGTRAQ', 'URL', 'MISC', 'MANDRIVA', 'OVAL', 'MLIST',
    'L0PHT', 'UBUNTU', 'AIXAPAR', 'REDHAT', 'EXPLOIT-DB', 'IBM', 'SGI',
    'APPLE', 'SF-INCIDENTS', 'APPLE', 'ERS', 'RSI', 'BINDVIEW', 'TRUSTIX',
    'CALDERA', 'ISS', 'DEBIAN', 'FARMERVENEMA', 'HPBUG', 'ATSTAKE', 'SREASON',
    'JVN', 'CERT', 'NAI', 'SUNALERT', 'TURBO', 'VULNWATCH', 'CONFIRM', 'HP',
    'SNI', 'SUSE'
]
EXTENSION_NUM = 300
github_list = ['MLIST', 'CONFIRM', 'MISC', 'URL', 'CONFIRM', 'XF', 'MISC']
DATE_COLUMNS = ["vulnerabilityAlerts","forks","issues","pullRequests","releases","stargazers"]
github_counter = 0


def safe_mkdir(dirname):
    with contextlib.suppress(FileExistsError):
        os.mkdir(dirname)


def ref_parser(ref_row):
    """

    :param ref_row: a reference to be parsed
    :return: a list of urls that might point to a commit
    """
    global github_counter
    refs = ref_row.split('|')
    ret_dict = {}
    has_github_ref = 0
    for ref in refs:
        with contextlib.suppress(ValueError):
            key, val = ref.split(":", 1)
            key = key.replace(' ', '')
            if "github" in val.lower():
                has_github_ref = 1
            if key in ret_dict:
                ret_dict[key].append(val)
            else:
                ret_dict[key] = [val]
    for ref_key in ref_keys:
        if ref_key not in ret_dict:
            ret_dict[ref_key] = []
    github_counter += has_github_ref
    return [ret_dict[x] for x in github_list] + [has_github_ref]


def handle_duplicate_key(key, ret_dict, val):
    found = False
    for i in range(1, EXTENSION_NUM):
        if f'{key}{i}' not in ret_dict:
            ret_dict[f'{key}{i}'] = val
            found = True
            break
    if not found:
        raise RuntimeError(f'{key} already in dict')


# token = open(r'C:\secrets\github_token.txt', 'r').read()
# g = Github(token)


def gather_pages(obj):
    obj_list = []
    obj.__requester.per_page = 100
    for i in range(0, obj.totalCount, 30):
        retry = True
        counter = 0
        while retry and counter < 100:
            try:
                retry = False
                for obj_instance in obj.get_page(i // 30):
                    obj_instance._completeIfNeeded = lambda: None
                    obj_list.append(obj_instance.raw_data)
            except Exception as e:
                print(obj)
                traceback.print_exc()
                counter += 1
    return obj_list


# todo For commits, get also additions and deletions
# todo number of subscriptions is not supported
# todo watchers and subscribers ??
attributes = [
    'commits', 'forks', 'comments', 'releases', 'events', 'issues', 'events',
    'pulls', 'pulls_comments', 'stargazers_with_dates'
]
attributes = ['stargazers_with_dates']
attributes = ['events']


def save_all_data(g, repo_name):
    repo = g.get_repo(repo_name)
    for attribute in attributes:
        print(f"\t{attribute}")
        Path(f"rawdata/{repo.name}").mkdir(parents=True, exist_ok=True)
        attr_func = repo.__getattribute__(f"get_{attribute}")
        with open(f'rawdata/{repo.name}/{attribute}.json', 'w') as fout:
            json.dump(gather_pages(attr_func()), fout, indent=4)


def yearly_preprocess(output_dir, repo_list):
    repo_dfs = []
    err_list = open("gh_yearly_errlist.txt", 'w')
    for repo_name, df in repo_list:
        if df[(df.type == 'VulnEvent')].empty:
            err_list.write(f"{repo_name}\n")
            continue
        day_df = pd.DataFrame()
        for col in df.type.unique():
            cur_type = df[(df.type == col)]
            if cur_type.empty:
                continue
            cur_df = pd.to_datetime(cur_type.created_at)
            cur_df = pd.DataFrame(cur_df).set_index('created_at')
            cur_df[col] = 1
            cur_df = cur_df.resample("D").sum()
            day_df = day_df.join(cur_df, how='outer')
            day_df = day_df.fillna(0)
        day_df.to_csv(f"{output_dir}/{repo_name.replace('/', '_')}.csv")


def parse_url(var):
    url = urlparse(var.lower())
    path = url.path + '/'
    commit_hash = ""
    if url.hostname != "github.com":
        return (None, None, None, None)
    if '/pull/' in path:
        if path.count('/') == 6:
            _, group, proj, pull, pull_num, commit, _ = path.split('/',
                                                                   maxsplit=6)
        else:
            _, group, proj, pull, pull_num, commit, commit_hash, _ = path.split(
                '/', maxsplit=7)
    else:
        _, group, proj, commit, commit_hash, _ = path.split('/', maxsplit=5)

    return group, proj, commit.replace(' ', ''), commit_hash


def extract_commits_from_projects_gh(cves):
    repo_commits = {}
    for _, row in cves[cves['has_github'] == 1].iterrows():
        for github_var in github_list:
            for var in row[github_var]:
                if '/commit' in var.lower():

                    group, proj, commit, commit_hash = parse_url(var)
                    if commit is None:
                        logger.debug(f"Unable to parse {var}")
                        continue

                    if commit in ['compare', 'blob']:
                        logger.debug(f"Unable to parse {var}")
                        continue

                    if commit not in ["commit", 'commits']:
                        logger.debug(f"Unable to parse {var}")
                        continue

                    proj_name = f"{group}/{proj}"
                    if proj_name not in repo_commits:
                        repo_commits[proj_name] = []

                    commit_hash = commit_hash.replace(' ', '')
                    commit_hash = commit_hash.replace('.patch', '')
                    commit_hash = commit_hash.replace('confirm:', '')
                    commit_hash = commit_hash[:40]
                    if commit_hash not in repo_commits[proj_name]:
                        repo_commits[proj_name].append(commit_hash)

    return repo_commits


def preprocess_dataframe(cves):
    cves = cves[~cves.ref.isna()]
    cves = cves.astype(str)
    new_ref_vals = zip(*cves['ref'].apply(ref_parser))
    for ref_val, name in zip(new_ref_vals, github_list + ['has_github']):
        cves[name] = ref_val
    return cves


datasets_foldername = "datasets"


def cve_preprocess(output_dir, cache_csv=False):
    logger.debug("Downloading CVE dataset")
    safe_mkdir(os.path.join(output_dir, datasets_foldername))
    if not cache_csv:
        cve_xml = "https://cve.mitre.org/data/downloads/allitems.csv"
        wget.download(cve_xml,
                      out=os.path.join(output_dir, datasets_foldername))

    cves = pd.read_csv(
        os.path.join(output_dir, datasets_foldername, "allitems.csv"),
        skiprows=11,
        encoding="ISO-8859-1",
        names=['cve', 'entry', 'desc', 'ref', 'assigned', 'un1', 'un2'],
        dtype=str)
    cves = preprocess_dataframe(cves)

    repo_commits = extract_commits_from_projects_gh(cves)
    with open(os.path.join(output_dir, 'repo_commits.json'), 'w') as fout:
        json.dump(repo_commits, fout, sort_keys=True, indent=4)


def graphql_preprocess(output_dir, project_name=None):
    with open(os.path.join(output_dir, 'repo_commits.json'), 'r') as fin:
        repo_commits = json.load(fin)

    repos = repo_commits.keys()
    for idx, repo in enumerate(repos):
        logger.debug(f"Processing {repo} ({idx}/{len(repos)})")

        if project_name is not None and not repo.endswith("/" + project_name):
            logger.error(f"Skipping {repo} since it has less that 10 CVEs")
            continue

        try:
            graphql.get_repo(output_dir, repo)
        except Exception as e:
            logger.error(f"Repository {repo} error at:" + repo +
                         traceback.format_exc())


def find_name(repo_commits, repo_name: str) -> str:
    return next(
        (key for key in repo_commits.keys() if key.endswith(f"/{repo_name}")),
        "")


def most_common(lst):
    return max(set(lst), key=lst.count) if lst else 0


def aggregate_all(output_dir):
    new_dfs = []

    with open(os.path.join(output_dir, 'repo_commits.json'), 'r') as fin:
        repo_commits = json.load(fin)

    # Getting graphql data

    print("[LOG] Getting graphql data:")
    for filename in os.listdir(os.path.join(output_dir,
                                            graphql.OUTPUT_DIRNAME))[:]:
        logger.debug(f"Getting graphql of {filename}")
        print(filename)
        df = pd.read_csv(
            os.path.join(output_dir, graphql.OUTPUT_DIRNAME, f"{filename}"))
        name = filename.split(".csv")[0]
        if df.empty:
            continue

        for col in DATE_COLUMNS:
            if not df[col].isnull().all():
                cur_df = pd.DataFrame()
                cur_df['created_at'] = pd.to_datetime(df[col].dropna())
                cur_df["type"] = col
                cur_df["name"] = name
                new_dfs.append(cur_df)

    # Getting gharchive data
    logger.debug("Getting gharchive data:")
    dfs = []
    for year in range(2015, 2020):
        logger.debug(f"gharchive of year {year}")
        dfs.append(
            pd.read_csv(
                os.path.join(output_dir, GITHUB_ARCHIVE_DIRNAME,
                             f'{year}.csv')))
    # adding vulnerabilities events

    logger.debug("Adding vulnerabilities events:")
    # Getting commit data

    repo_commit_df_lst = []
    for repo, vuln_commits in list(repo_commits.items())[:]:
        logger.debug(f"Aggregating repo {repo}")
        repo_real_name = repo.replace('/', '_')
        with open(
                os.path.join(output_dir, json_commit_directory,
                             f"{repo_real_name}.json"), 'r') as fin:
            all_commits = json.load(fin)

        for commit in all_commits:
            if commit[0] in vuln_commits:
                commit.append(1)
            else:
                commit.append(0)
            commit.append(repo_real_name)
            commit.append("Commit")

        repo_commit_df = pd.DataFrame(all_commits,
                                      columns=[
                                          'Hash', 'created_at', "Add", "Del",
                                          "Files", "Vuln", "name", "type"
                                      ])

        repo_commit_df_lst.append(
            repo_commit_df)  # df['Time'] = pd.to_datetime(df['Time'])

    logger.debug("Concatenating dataframes")
    df = pd.concat(dfs + repo_commit_df_lst + new_dfs)

    logger.debug("Replacing / with _")
    df.name = df.name.str.replace("/", "_")

    logger.debug("Grouping Dataframes")
    repo_list = list(df.groupby('name'))

    logger.debug("saving data to parquets")
    safe_mkdir(os.path.join(output_dir, gh_cve_dir))
    for repo_name, df in repo_list:
        logger.debug(f"Saving {repo_name} to parquet")
        df.to_csv(
            os.path.join(output_dir, gh_cve_dir,
                         f"{repo_name.replace('/', '_')}.csv"))


def extract_commits_from_projects(output_dir):

    safe_mkdir(os.path.join(output_dir, commit_directory))
    safe_mkdir(os.path.join(output_dir, json_commit_directory))

    with open(os.path.join(output_dir, 'repo_commits.json'), 'r') as fin:
        repo_commits = json.load(fin)

    for repo_name in repo_commits.keys():
        logger.debug(f"Processing {repo_name}")
        author, repo = repo_name.split("/")
        repo_directory = f"{author}_{repo}"
        commit_cur_dir = os.path.join(output_dir, commit_directory,
                                      repo_directory)
        repo_url = f"https://github.com/{repo_name}.git"

        subprocess.run(f"git clone --mirror {repo_url} {commit_cur_dir}",
                       shell=True)

        subprocess.run(f"git -C {commit_cur_dir} fetch --unshallow",
                       shell=True)
        commit_abs_path = os.path.abspath(commit_cur_dir)
        gitlog = run_git_log(commit_abs_path)
        jsons = git2jsons(gitlog)

        commits = json.loads(jsons)
        res = []
        timezones = []
        for commit in commits:
            # gathering timezones from commits
            tz = commit['committer']['timezone']
            tz = int(tz[:-2]) + int(tz[-2:]) / 60.0
            timezones.append(tz)

            adds, dels = 0, 0
            for change in commit['changes']:
                if change is not None:
                    if type(change[0]) == int:
                        adds += change[0]
                    if type(change[1]) == int:
                        dels += change[1]

            time = datetime.utcfromtimestamp(
                commit['committer']['date']).strftime('%Y-%m-%d %H:%M:%S')

            res.append(
                (commit['commit'], time, adds, dels, len(commit['changes'])))

        avg_timezone = most_common(timezones)
        with open(
                os.path.join(output_dir, json_commit_directory,
                             f'{author}_{repo}.json'), 'w') as fout:
            json.dump(res, fout, indent=4)
        safe_mkdir(os.path.join(output_dir, timezone_directory))
        with open(
                os.path.join(output_dir, timezone_directory,
                             f'{author}_{repo}.json'), 'w') as fout:
            fout.write(str(avg_timezone))


def metadata_preprocess(output_dir):
    all_langs = set()
    repos = {}

    with open(os.path.join(output_dir, 'repo_commits.json'), 'r') as fin:
        repo_commits = json.load(fin)

    for repo_name in repo_commits.keys():
        logger.debug(f"Processing {repo_name}")
        author, repo = repo_name.split("/")
        repo_metadata = graphql.get_commit_metadata(author, repo)
        if not repo_metadata:
            continue
        all_langs = all_langs.union(set(repo_metadata['languages_edges']))
        repos[repo_name] = repo_metadata

        # res['languages_edges']='|'.join(list(map(lambda lang: lang['node']['name'],res['languages_edges'])))

    with open(os.path.join(output_dir, 'repo_metadata.json'), 'w') as mfile:
        json.dump(repos, mfile)


def main(graphql=False,
         cve=False,
         metadata=False,
         commits=False,
         aggregate=False,
         all=False,
         output_dir="output"):
    if all or cve:
        cve_preprocess(output_dir)
    if all or graphql:
        graphql_preprocess(output_dir)
    if all or metadata:
        metadata_preprocess(output_dir)
    if all or commits:
        extract_commits_from_projects(output_dir)
    if all or aggregate:
        aggregate_all(output_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Detects hidden cves')
    parser.add_argument("--cve",
                        action="store_true",
                        help="Runs cve preprocessing")
    parser.add_argument("--graphql",
                        action="store_true",
                        help="Runs graphql preprocessing")
    parser.add_argument("--metadata",
                        action="store_true",
                        help="Stores metadata of repository")
    parser.add_argument("--commits",
                        action="store_true",
                        help="acquire all commits")
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help="Runs aggregation with graphql and gharpchive data")
    parser.add_argument("--all",
                        action="store_true",
                        help="Run all preprocessing steps")
    parser.add_argument("-o", "--output-dir", action="store", default="data")
    args = parser.parse_args()

    main(graphql=args.graphql,
         cve=args.cve,
         metadata=args.metadata,
         commits=args.commits,
         aggregate=args.aggregate,
         all=args.all,
         output_dir=args.output_dir)
