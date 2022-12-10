import argparse
import json
import os
import subprocess
import traceback

import pandas as pd
import wget
import graphql

from pathlib import Path
from tqdm import tqdm
from urllib.parse import urlparse
from git2json import *
from datetime import datetime



GITHUB_ARCHIVE_DIRNAME = "gharchive"
OUTPUT_DIR = "gh_cve_proccessed"

LOG_GRAPHQL = "graphql_errlist.txt"
LOG_AGGR_ALL = 'gharchive_errlist.txt'
key_list = set()
err_counter = 0
ref_keys = ['ASCEND', 'ENGARDE', 'VIM', 'ERS', 'ATSTAKE', 'JVNDB', 'SLACKWARE', 'ENGARDE', 'OPENBSD',
            'CIAC', 'IBM', 'SUNALERT', 'FARMERVENEMA', 'XF', 'ALLAIRE', 'VULN-DEV', 'MSKB', 'VULNWATCH',
            'AIXAPAR', 'CERT-VN', 'NTBUGTRAQ', 'XF', 'SUSE', 'CONECTIVA', 'SEKURE', 'MISC', 'MSKB',
            'SUNBUG', 'TURBO', 'VUPEN', 'BUGTRAQ', 'BUGTRAQ', 'DEBIAN', 'SCO', 'MS', 'IDEFENSE', 'MLIST',
            'INFOWAR', 'SECUNIA', 'FULLDISC', 'SUN', 'KSRT', 'HP', 'BID', 'EL8', 'MANDRAKE', 'IMMUNIX',
            'SECTRACK', 'VULN-DEV', 'CONFIRM', 'GENTOO', 'SECTRACK', 'EL8', 'VUPEN', 'CERT', 'FREEBSD',
            'HERT', 'L0PHT', 'BID', 'CONECTIVA', 'SREASONRES', 'SCO', 'FEDORA', 'NAI', 'AUSCERT',
            'ISS', 'COMPAQ', 'NETECT', 'SUNBUG', 'CHECKPOINT', ' 1.23.1)', 'WIN2KSEC', 'BEA', 'EXPLOIT-DB',
            'KSRT', 'MANDRAKE', 'FRSIRT', 'JVN', 'RSI', 'NETBSD', 'AUSCERT', 'OPENPKG', 'OPENBSD',
            'CERT-VN', 'SUN', 'SECUNIA', 'VIM', 'GENTOO', 'REDHAT', 'MS',
            'COMPAQ', 'OVAL', 'CALDERA', 'FEDORA', 'FREEBSD', 'CISCO', 'CISCO', 'WIN2KSEC', 'MANDRIVA', 'OSVDB',
            'UBUNTU', 'EEYE', 'BEA', 'IDEFENSE', 'NETBSD', 'SGI', 'SREASON', 'OSVDB', 'CIAC', 'BINDVIEW',
            'FULLDISC', 'NTBUGTRAQ', 'URL', 'MISC', 'MANDRIVA', 'OVAL', 'MLIST', 'L0PHT', 'UBUNTU',
            'AIXAPAR', 'REDHAT', 'EXPLOIT-DB', 'IBM', 'SGI', 'APPLE', 'SF-INCIDENTS', 'APPLE', 'ERS',
            'RSI', 'BINDVIEW', 'TRUSTIX', 'CALDERA', 'ISS', 'DEBIAN', 'FARMERVENEMA', 'HPBUG',
            'ATSTAKE', 'SREASON', 'JVN', 'CERT', 'NAI', 'SUNALERT', 'TURBO', 'VULNWATCH', 'CONFIRM',
            'HP', 'SNI', 'SUSE']
EXTENSION_NUM = 300
github_list = ['MLIST', 'CONFIRM', 'MISC', 'URL', 'CONFIRM', 'XF', 'MISC']

github_counter = 0




def safe_mkdir(dirname):
    try:
        os.mkdir(dirname)
    except FileExistsError:
        pass


def ref_parser(ref_row):
    """

    :param ref_row: a reference to be parsed
    :return: a list of urls that might point to a commit
    """
    global github_counter
    refs = ref_row.split('|')
    ret_dict = dict()
    has_github_ref = 0
    for ref in refs:
        try:
            key, val = ref.split(":", 1)
            key = key.replace(' ', '')
            if "github" in val.lower():
                has_github_ref = 1
            if key in ret_dict:
                ret_dict[key].append(val)
            else:
                ret_dict[key] = [val]
        except ValueError:
            pass

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
        raise BaseException(f'{key} already in dict')


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
                for obj_instance in obj.get_page(i // 30):
                    obj_instance._completeIfNeeded = lambda: None
                    obj_list.append(obj_instance.raw_data)
                    retry = False
            except Exception as e:
                print(obj)
                traceback.print_exc()
                counter += 1
                pass
    return obj_list


# todo For commits, get also additions and deletions
# todo number of subscriptions is not supported
# todo watchers and subscribers ??
attributes = ['commits', 'forks', 'comments', 'releases', 'events', 'issues', 'events', 'pulls', 'pulls_comments',
              'stargazers_with_dates']
attributes = ['stargazers_with_dates']
attributes = ['events']


def save_all_data(repo_name):
    repo = g.get_repo(repo_name)
    for attribute in attributes:
        print(f"\t{attribute}")
        Path(f"rawdata/{repo.name}").mkdir(parents=True, exist_ok=True)
        attr_func = repo.__getattribute__(f"get_{attribute}")
        with open(f'rawdata/{repo.name}/{attribute}.json', 'w') as fout:
            json.dump(gather_pages(attr_func()), fout, indent=4)


def yearly_preprocess():
    import tqdm
    repo_dfs = []
    err_list = open("gh_yearly_errlist.txt", 'w')
    for repo_name, df in tqdm.tqdm(repo_list):
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
        day_df.to_csv(f"{OUTPUT_DIR}/{repo_name.replace('/', '_')}.csv")


def extract_commits_from_projects_gh(cves):
    repo_commits = dict()
    for index, row in tqdm(cves[cves['has_github'] == 1].iterrows()):
        for github_var in github_list:
            for var in row[github_var]:
                if '/commit' in var.lower():
                    # try:
                    url = urlparse(var.lower())
                    path = url.path + '/'
                    commit_hash = ""
                    if url.hostname != "github.com":
                        continue
                    if '/pull/' in path:
                        if path.count('/') == 6:
                            _, group, proj, pull, pull_num, commit, rest = path.split(
                                '/', maxsplit=6)
                        else:
                            _, group, proj, pull, pull_num, commit, commit_hash, rest = path.split('/',
                                                                                                   maxsplit=7)
                    else:
                        _, group, proj, commit, commit_hash, rest = path.split(
                            '/', maxsplit=5)

                    commit = commit.replace(' ', '')
                    if commit == 'compare' or commit == 'blob':
                        print(f"Unable to parse {path}")
                        continue
                    assert commit == "commit" or commit == 'commits'
                    if group in repo_dict:
                        group = repo_dict[group]
                    proj_name = f"{group}/{proj}"
                    if proj_name not in repo_commits:
                        repo_commits[proj_name] = []
                    # todo use pull number also
                    # todo find all projects with this names and check if the have this commit hash
                    commit_hash = commit_hash.replace(' ', '')
                    commit_hash = commit_hash.replace('.patch', '')
                    commit_hash = commit_hash.replace('confirm:', '')
                    print(index)
                    try:
                        repo_commits[proj_name].append(
                            (commit_hash, graphql.get_date_for_commit(proj_name, commit_hash)))
                    except graphql.RepoNotFoundError:
                        proj_name, date = graphql.get_date_for_alternate_proj_commit(
                            proj_name, commit_hash)
                        if proj_name is not None:
                            if proj_name not in repo_commits:
                                repo_commits[proj_name] = []
                            repo_commits[proj_name].append((commit_hash, date))
    return repo_commits


def preprocess_dataframe(cves):
    cves = cves[~cves.ref.isna()]
    cves = cves.astype(str)
    new_ref_vals = zip(*cves['ref'].apply(ref_parser))
    for ref_val, name in zip(new_ref_vals, github_list + ['has_github']):
        cves[name] = ref_val
    return cves


def cve_preprocess(use_cached=False, dont_extract=False):
    cve_xml = "https://cve.mitre.org/data/downloads/allitems.csv"
    xml_file_name = cve_xml
    if not use_cached:
        wget.download(xml_file_name, out='datasets')

    if not dont_extract:
        cves = pd.read_csv('datasets/allitems.csv', skiprows=11, encoding="ISO-8859-1",
                           names=['cve', 'entry', 'desc', 'ref', 'assigned', 'un1', 'un2'])
        cves = preprocess_dataframe(cves)

        repo_commits = extract_commits_from_projects(cves)
        with open(f'repo_commits.json', 'w') as fout:
            json.dump(repo_commits, fout, sort_keys=True, indent=4)


def graphql_preprocess(project_name=None):
    err_list = open(LOG_GRAPHQL, 'a')
    with open(f'repo_commits.json', 'r') as fin:
        repo_commits = json.load(fin)
    idx = 0
    repos = repo_commits.keys()
    for repo in list(repos)[:]:
        if repo.replace("/","_") in graphql.less_than_10_vulns:
            print(f"Skipping {repo}")
            continue

        print(idx, repo)
        idx += 1

        if project_name is not None and not repo.endswith("/" + project_name):
            continue

        try:
            graphql.get_repo(repo, idx)
        except Exception as e:
            print("Error at"+repo+traceback.format_exc())
            err_list.write(f"{repo}:{traceback.format_exc()}\n")

    err_list.close()


with open(f'repo_commits.json', 'r') as fin:
    repo_commits = json.load(fin)


def find_name(repo_name: str) -> str:
    for key in repo_commits.keys():
        if key.endswith("/" + repo_name):
            return key
    return ""

def most_common(lst):
    if not lst:
        return 0
    return max(set(lst), key=lst.count)

def aggregate_all():
    new_dfs = []
    tqdm.pandas()

    # Getting graphql data

    print("[LOG] Getting graphql data:")
    for filename in os.listdir(graphql.OUTPUT_DIRNAME)[:]:
        print(filename)
        df = pd.read_csv(f"{graphql.OUTPUT_DIRNAME}/{filename}")
        name = filename.split(".csv")[0]
        if df.empty:
            continue

        for col in ['vulnerabilityAlerts', 'forks', 'issues', 'pullRequests', 'releases', 'stargazers']:
            if not df[col].isnull().all():
                cur_df = pd.DataFrame()
                cur_df['created_at'] = pd.to_datetime(df[col].dropna())
                cur_df["type"] = col
                cur_df["name"] = name
                new_dfs.append(cur_df)


    
    # Getting gharchive data
    print("[LOG] Getting gharchive data:")
    dfs = []
    for year in range(2015, 2020):
        print(year)
        dfs.append(pd.read_csv(f'{GITHUB_ARCHIVE_DIRNAME}/{year}.csv'))

    # adding vulnerabilities events

    print("[LOG] Adding vulnerabilities events:")
    # Getting commit data
       
    repo_commit_df_lst = []    
    for repo, dates in list(repo_commits.items())[:]:
        print(repo)
        repo_real_name = repo.replace('/','_')
        with open(f"commits/{repo_real_name}.json", 'r') as fin:
            all_commits = json.load(fin)

        vuln_commits = [date[0] for date in dates]
        for commit in all_commits:
            if commit[0] in vuln_commits:
                commit.append(1)
            else:
                commit.append(0)
            commit.append(repo_real_name)
            commit.append("Commit")

        repo_commit_df = pd.DataFrame(all_commits, columns = ['Hash', 'created_at',"Add","Del","Files","Vuln","name","type"])
        repo_commit_df_lst.append(repo_commit_df) # df['Time'] = pd.to_datetime(df['Time'])

    df = pd.concat(dfs + repo_commit_df_lst + new_dfs).progress_apply(lambda x: x)
    df.name = df.name.str.replace("/", "_")
    repos = df.name.unique()
    repo_list = [repo for repo in df.groupby('name')]

    repo_dfs = []
    i = 0
    err_list = open(LOG_AGGR_ALL, 'a')
    print("[LOG] saving data to csvs")
    for repo_name, df in repo_list:
        df.to_csv(f"{OUTPUT_DIR}/{repo_name.replace('/', '_')}.csv")
    err_list.close()


def extract_commits_from_projects():
    try:
        os.mkdir("commits")
    except FileExistsError:
        pass

    with open(f'repo_commits.json', 'r') as fin:
        repo_commits = json.load(fin)
    all_repos = dict()
    time_zones = dict()

    for repo_name in repo_commits.keys():
        print(repo_name)
        author,repo = repo_name.split("/")
        repo_url = f"https://github.com/{repo_name}.git"

        subprocess.run(
            f"git clone --mirror {repo_url} commits\\{author}_{repo}", shell=True)

        subprocess.run(f"git -C commits\\{author}_{repo} fetch --unshallow", shell=True)
        a = os.path.abspath(f"commits\\{author}_{repo}")
        gitlog = run_git_log(a)
        jsons = git2jsons(gitlog)

        commits = json.loads(jsons)
        res = []
        timezones = []
        for commit in commits:
            # gathering timezones from commits
            tz = commit['committer']['timezone']
            tz = int(tz[:-2])+int(tz[-2:])/60.0
            timezones.append(tz)

            adds, dels = 0,0
            for change in commit['changes']:
                if type(change[0]) == int:
                    adds += change[0]
                if type(change[1])== int:
                    dels += change[1]

            time = datetime.utcfromtimestamp(commit['committer']['date']).strftime('%Y-%m-%d %H:%M:%S')
            
            res.append((commit['commit'],time, adds, dels,len(commit['changes'])))

        avg_timezone = most_common(timezones)
        with open(f'commits/{author}_{repo}.json', 'w') as fout:
                json.dump(res, fout, indent=4)
        safe_mkdir('timezones')
        print(avg_timezone)
        with open(f'timezones/{author}_{repo}.txt', 'w') as fout:
            fout.write(str(avg_timezone))


def metadata_preprocess():
    all_langs = set()
    repos = dict()
    for repo_name in tqdm(repo_commits.keys()):
        author,repo = repo_name.split("/")
        repo_metadata = graphql.get_commit_metadata(author,repo)
        if not repo_metadata:
            continue
        all_langs = all_langs.union(set(repo_metadata['languages_edges']))
        repos[repo_name] = repo_metadata

        # res['languages_edges']='|'.join(list(map(lambda lang: lang['node']['name'],res['languages_edges'])))

    with open('repo_metadata.json','w') as mfile:
        json.dump(repos,mfile)
    

def main(graphql=False, cve=False,metadata=False, commits=False, aggregate=False):

    if cve:
        cve_preprocess()
    if graphql:
        graphql_preprocess()
    if metadata:
        metadata_preprocess()
    if commits:
        extract_commits_from_projects()
    if aggregate:
        aggregate_all()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detects hidden cves')
    parser.add_argument("--graphql", action="store_true",
                        help="Runs graphql preprocessing")
    parser.add_argument("--cve", action="store_true",
                        help="Runs cve preprocessing")
    parser.add_argument("--metadata", action="store_true",help="Stores metadata of repository")
    parser.add_argument("--commits",action="store_true",help="acquire all commits")
    parser.add_argument("--aggregate", action="store_true",
                        help="Runs aggregation with graphql and gharpchive data")
    args = parser.parse_args()

    main(graphql=args.graphql, cve=args.cve,metadata=args.metadata, commits=args.commits, aggregate=args.aggregate)
