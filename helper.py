import contextlib
from functools import wraps
from time import time
import itertools
import requests
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import array
from sklearn.metrics import precision_recall_fscore_support as f_score
from sklearn.metrics import accuracy_score as a_score
import os
import argparse
import enum
import json
from data_collection.graphql import all_langs
from dateutil import parser
from collections import Counter
import numpy as np


class Repository:

    def __init__(self):
        self.vuln_lst = []
        self.benign_lst = []
        self.vuln_details = []
        self.benign_details = []
        self.column_names = []
        self.file = ""
        self.metadata = None

    def pad_repo(self, to_pad=None):
        padded_vuln_all, padded_benign_all = [], []
        if to_pad is None:
            to_pad = max(max(Counter([v.shape[0] for v in self.vuln_lst])),
                         max(Counter([v.shape[0] for v in self.benign_lst])))

        padded_vuln_all.extend(
            np.pad(vuln, ((to_pad - vuln.shape[0], 0), (0, 0)))
            for vuln in self.vuln_lst)

        padded_benign_all.extend(
            np.pad(benign, ((to_pad - benign.shape[0], 0), (0, 0)))
            for benign in self.benign_lst)

        self.vuln_lst = np.nan_to_num(np.array(padded_vuln_all))
        self.benign_lst = np.nan_to_num(np.array(padded_benign_all))

    def get_all_lst(self):
        X = np.concatenate([self.vuln_lst, self.benign_lst])
        y = len(self.vuln_lst) * [1] + len(self.benign_lst) * [0]
        return X, y

    def get_num_of_vuln(self):
        return len(self.vuln_lst)

    def get_all_details(self):
        return np.concatenate([self.vuln_details, self.benign_details])


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError(
                "type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, enum.Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


def normalize(time_series_feature):
    if time_series_feature.max() - time_series_feature.min() == 0:
        return time_series_feature
    return (time_series_feature - time_series_feature.min()) / (
        time_series_feature.max() - time_series_feature.min())


def split_sequence(sequence, n_steps):
    X, y = [], []
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


def draw_timeline(name, vulns, first_date, last_date):
    dates = vulns
    dates += [first_date]
    dates += [last_date]

    values = [1] * len(dates)
    values[-1] = 2
    values[-2] = 2

    string_dates = pd.to_datetime(dates).strftime("%Y-%m-%d %H:%M:%S").tolist() 
    
    fig, ax = plt.subplots(figsize=(6, 1))
    
    ax.scatter(string_dates.tolist(), [1] * len(string_dates), c=values, marker='s', s=100)

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
    plt.show()


def find_best_acc(X_test, y_test, model, verbose=0):
    best_acc = 0
    best_model = None
    y_pred = model.predict(X_test, verbose=verbose)
    best_thresh = 0
    for i in range(100):
        acc = a_score(y_test, (y_pred > i / 100).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_thresh = i
    return best_acc, best_thresh


def find_best_f1(X_test, y_test, model):
    max_f1 = 0
    thresh = 0
    best_y = 0
    pred = model.predict(X_test)
    for i in range(100):
        y_predict = (pred.reshape(-1) > i / 100).astype(int)
        precision, recall, fscore, support = f_score(y_test,
                                                     y_predict)
        if len(fscore) == 1:
            return 0, 0, 0
        cur_f1 = fscore[1]
        # print(i,cur_f1)
        if cur_f1 > max_f1:
            max_f1 = cur_f1
            best_y = y_predict
            thresh = i / 100
    return max_f1, thresh, best_y


def find_best_accuracy(X_test, y_test, model):
    max_score = 0
    thresh = 0
    best_y = 0
    pred = model.predict(X_test)
    for i in range(100):
        y_predict = (pred.reshape(-1) > i / 100).astype(int)
        score = a_score(y_test.astype(float), y_predict)
        # print(i,cur_f1)
        if score > max_score:
            max_score = score
            best_y = y_predict
            thresh = i / 100
    return max_score, thresh, best_y


def generator(feat, labels):
    pairs = [(x, y) for x in feat for y in labels]
    cycle_pairs = itertools.cycle(pairs)
    for a, b in pairs:
        yield np.array([a]), np.array([b])
    return


def find_threshold(model, x_train_scaled):
    import tensorflow as tf
    reconstructions = model.predict(x_train_scaled)
    # provides losses of individual instances
    reconstruction_errors = tf.keras.losses.msle(reconstructions,
                                                 x_train_scaled)
    return np.mean(reconstruction_errors.numpy()) + np.std(
        reconstruction_errors.numpy())


def get_predictions(model, x_test_scaled, threshold):
    import tensorflow as tf

    predictions = model.predict(x_test_scaled)
    # provides losses of individual instances
    errors = tf.keras.losses.msle(predictions, x_test_scaled)
    # 0 = anomaly, 1 = normal
    anomaly_mask = pd.Series(errors) > threshold
    return anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)


token = open(r'C:\secrets\github_token.txt', 'r').read()
headers = {"Authorization": "token " + token}

commits_between_dates = """
{{
    repository(owner: "{0}", name:"{1}") {{
        object(expression: "{2}") {{
            ... on Commit {{
                history(first: 100, since: "{3}", until: "{4}") {{
                    nodes {{
                      commitUrl,
                      message
                    }}
                }}
            }}
    }}
  }}
}}




"""


def run_query(query, ignore_errors=False):
    counter = 0
    while True:
        request = requests.post('https://api.github.com/graphql',
                                json={'query': query},
                                headers=headers)
        if request.status_code == 200:
            return request.json()
        elif request.status_code == 502:
            raise RuntimeError(
                f"Query failed to run by returning code of {request.status_code}. {request}"
            )

        else:
            request_json = request.json()
            if "errors" in request_json and (
                    "timeout" in request_json["errors"][0]["message"]
                    or request_json["errors"]["type"] == 'RATE_LIMITED'):

                print("Waiting for an hour")
                print(request, request_json)
                counter += 1
                if counter < 6:
                    time.sleep(60 * 60)
                    continue
                break

            err_string = f"Query failed to run by returning code of {request.status_code}. {query}"

            if ignore_errors:
                print(err_string)
            else:
                raise RuntimeError(err_string)


def safe_mkdir(dirname):
    with contextlib.suppress(FileExistsError):
        os.mkdir(dirname)


def timing(f):

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return wrap


bool_metadata = [
    'owner_isVerified', 'owner_isHireable', 'owner_isGitHubStar',
    "owner_isCampusExpert", "owner_isDeveloperProgramMember",
    'owner_isSponsoringViewer', 'owner_isSiteAdmin', 'isInOrganization',
    'hasIssuesEnabled', 'hasWikiEnabled', 'isMirror',
    'isSecurityPolicyEnabled', 'diskUsage', 'owner_isEmployee'
]


def add_metadata(data_path,
                 all_metadata,
                 cur_repo,
                 file,
                 repo_holder: Repository):

    cur_metadata = all_metadata[file.replace("_", "/", 1).lower()]
    if repo_holder is not None:
        repo_holder.metadata = cur_metadata

    for key in bool_metadata:
        cur_repo[key] = 0

    handle_nonbool_metadata(cur_repo, cur_metadata)
    handle_timezones(data_path, cur_repo, file, repo_holder)

    return cur_repo


def handle_nonbool_metadata(cur_repo, cur_metadata):
    for key, value in cur_metadata.items():
        if key == "languages_edges":
            for lang in all_langs:
                cur_repo[lang] = 0
            for lang in value:
                if lang in all_langs:
                    cur_repo[lang] = 1

        elif key == "createdAt":  # this is probably lower performance
            for i in range(2000, 2023):
                cur_repo[f"repo_creation_data_{str(i)}"] = 0
            if f"repo_creation_data_{str(parser.parse(value).year)}" not in cur_repo.columns:
                raise RuntimeError(f"not exist {value}")
            cur_repo[f"repo_creation_data_{str(parser.parse(value).year)}"] = 1

        elif key == "fundingLinks":
            cur_repo[key] = len(value)

        elif key in bool_metadata:
            cur_repo[key] = int(value) if value else 0
        elif key in [
                'primaryLanguage_name', 'primaryLanguage', "owner_company"
        ]:
            continue

        else:
            if key not in cur_repo.columns:
                print(key)


def handle_timezones(data_path, cur_repo, file, repo_holder):
    with open(os.path.join(data_path, "timezones", file + ".json"), 'r') as f:
        timezone = int(float(f.read()))
    if repo_holder is not None:
        repo_holder.metadata["timezone"] = timezone
    for tz in range(-12, 15):
        cur_repo[f"timezone_{str(tz)}"] = 0

    cur_repo[f"timezone_{timezone}"] = 1


all_langs = [
    '1C Enterprise', 'AGS Script', 'AIDL', 'AMPL', 'ANTLR', 'API Blueprint',
    'ASL', 'ASP', 'ASP.NET', 'ActionScript', 'Ada', 'Agda', 'Alloy',
    'AngelScript', 'ApacheConf', 'Apex', 'AppleScript', 'Arc', 'AspectJ',
    'Assembly', 'Asymptote', 'Augeas', 'AutoHotkey', 'AutoIt', 'Awk', 'BASIC',
    'Ballerina', 'Batchfile', 'Berry', 'Bicep', 'Bikeshed', 'BitBake', 'Blade',
    'BlitzBasic', 'Boogie', 'Brainfuck', 'Brightscript', 'C', 'C#', 'C++',
    'CMake', 'COBOL', 'CSS', 'CUE', 'CWeb', 'Cadence', "Cap'n Proto", 'Ceylon',
    'Chapel', 'Charity', 'ChucK', 'Clarion', 'Classic ASP', 'Clean', 'Clojure',
    'Closure Templates', 'CodeQL', 'CoffeeScript', 'ColdFusion', 'Common Lisp',
    'Common Workflow Language', 'Coq', 'Cuda', 'Cython', 'D',
    'DIGITAL Command Language', 'DM', 'DTrace', 'Dart', 'Dhall', 'Dockerfile',
    'Dylan', 'E', 'ECL', 'EJS', 'Eiffel', 'Elixir', 'Elm', 'Emacs Lisp',
    'EmberScript', 'Erlang', 'Euphoria', 'F#', 'F*', 'FLUX', 'Fancy', 'Faust',
    'Filebench WML', 'Fluent', 'Forth', 'Fortran', 'FreeBasic', 'FreeMarker',
    'GAP', 'GCC Machine Description', 'GDB', 'GDScript', 'GLSL', 'GSC',
    'Game Maker Language', 'Genshi', 'Gherkin', 'Gnuplot', 'Go', 'Golo',
    'Gosu', 'Groff', 'Groovy', 'HCL', 'HLSL', 'HTML', 'Hack', 'Haml',
    'Handlebars', 'Haskell', 'Haxe', 'Hy', 'IDL', 'IGOR Pro', 'Inform 7',
    'Inno Setup', 'Ioke', 'Isabelle', 'Jasmin', 'Java', 'JavaScript',
    'JetBrains MPS', 'Jinja', 'Jolie', 'Jsonnet', 'Julia', 'Jupyter Notebook',
    'KRL', 'Kotlin', 'LLVM', 'LSL', 'Lasso', 'Latte', 'Less', 'Lex', 'Limbo',
    'Liquid', 'LiveScript', 'Logos', 'Lua', 'M', 'M4', 'MATLAB', 'MAXScript',
    'MLIR', 'MQL4', 'MQL5', 'Macaulay2', 'Makefile', 'Mako', 'Mathematica',
    'Max', 'Mercury', 'Meson', 'Metal', 'Modelica', 'Modula-2', 'Modula-3',
    'Module Management System', 'Monkey', 'Moocode', 'MoonScript', 'Motoko',
    'Mustache', 'NASL', 'NSIS', 'NewLisp', 'Nextflow', 'Nginx', 'Nim', 'Nit',
    'Nix', 'Nu', 'OCaml', 'Objective-C', 'Objective-C++', 'Objective-J',
    'Open Policy Agent', 'OpenEdge ABL', 'PEG.js', 'PHP', 'PLSQL', 'PLpgSQL',
    'POV-Ray SDL', 'Pan', 'Papyrus', 'Pascal', 'Pawn', 'Perl', 'Perl 6',
    'Pike', 'Pony', 'PostScript', 'PowerShell', 'Processing', 'Procfile',
    'Prolog', 'Promela', 'Pug', 'Puppet', 'PureBasic', 'PureScript', 'Python',
    'QML', 'QMake', 'R', 'RAML', 'REXX', 'RPC', 'RPGLE', 'RUNOFF', 'Racket',
    'Ragel', 'Ragel in Ruby Host', 'Raku', 'ReScript', 'Reason', 'Rebol',
    'Red', 'Redcode', 'RenderScript', 'Rich Text Format', 'Riot',
    'RobotFramework', 'Roff', 'RouterOS Script', 'Ruby', 'Rust', 'SAS', 'SCSS',
    'SMT', 'SQLPL', 'SRecode Template', 'SWIG', 'Sage', 'SaltStack', 'Sass',
    'Scala', 'Scheme', 'Scilab', 'Shell', 'ShellSession', 'Sieve', 'Slice',
    'Slim', 'SmPL', 'Smali', 'Smalltalk', 'Smarty', 'Solidity', 'SourcePawn',
    'Stan', 'Standard ML', 'Starlark', 'Stata', 'StringTemplate', 'Stylus',
    'SuperCollider', 'Svelte', 'Swift', 'SystemVerilog', 'TLA', 'TSQL', 'Tcl',
    'TeX', 'Tea', 'Terra', 'Thrift', 'Turing', 'Twig', 'TypeScript',
    'UnrealScript', 'VBA', 'VBScript', 'VCL', 'VHDL', 'Vala',
    'Velocity Template Language', 'Verilog', 'Vim Snippet', 'Vim script',
    'Visual Basic', 'Visual Basic .NET', 'Volt', 'Vue', 'WebAssembly', 'Wren',
    'X10', 'XProc', 'XQuery', 'XS', 'XSLT', 'Xtend', 'YARA', 'Yacc', 'Yul',
    'Zeek', 'Zig', 'eC', 'jq', 'kvlang', 'mupad', 'nesC', 'q', 'sed', 'xBase'
]
