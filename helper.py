import itertools
import requests
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import array
from sklearn.metrics import precision_recall_fscore_support as f_score
from sklearn.metrics import accuracy_score as a_score

import argparse
import enum


class EnumAction(argparse.Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
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
    return (time_series_feature - time_series_feature.min()) / (time_series_feature.max() - time_series_feature.min())


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


def draw_timeline(name, vulns, first_date, last_date):
    dates = vulns
    dates += [first_date]
    dates += [last_date]

    values = [1] * len(dates)
    values[-1] = 2
    values[-2] = 2

    X = pd.to_datetime(dates)
    fig, ax = plt.subplots(figsize=(6, 1))
    ax.scatter(X, [1] * len(X), c=values,
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
    plt.show()


def find_best_f1(X_test, y_test, model):
    max_f1 = 0
    thresh = 0
    best_y = 0
    pred = model.predict(X_test)
    for i in range(100):
        y_predict = (pred.reshape(-1) > i / 100).astype(int)
        precision, recall, fscore, support = f_score(y_test, y_predict, zero_division=0)
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
        y_predict = (pred.reshape(-1) > i / 1000).astype(int)
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
    reconstruction_errors = tf.keras.losses.msle(reconstructions, x_train_scaled)
    # threshold for anomaly scores
    threshold = np.mean(reconstruction_errors.numpy()) + np.std(reconstruction_errors.numpy())
    return threshold


def get_predictions(model, x_test_scaled, threshold):
    import tensorflow as tf

    predictions = model.predict(x_test_scaled)
    # provides losses of individual instances
    errors = tf.keras.losses.msle(predictions, x_test_scaled)
    # 0 = anomaly, 1 = normal
    anomaly_mask = pd.Series(errors) > threshold
    preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
    return preds


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

def run_query(query):
    counter = 0;
    while True:
        request = requests.post('https://api.github.com/graphql', json={'query': query}, headers=headers)
        if request.status_code == 200:
            return request.json()
        elif request.status_code == 502:
            raise Exception(
                "Query failed to run by returning code of {}. {}".format(request.status_code, request, query))
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

            raise Exception("Query failed to run by returning code of {}. {}".format(request.status_code, query))


