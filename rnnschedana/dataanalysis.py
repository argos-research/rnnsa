from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import matplotlib.ticker as mplticker
import matplotlib.colors as mplcolors
from matplotlib.collections import LineCollection
import IPython as ip

import datageneration as gen
import schedulerproxy as sch
import samples
import interface
from mongomonitor import MongoMonitor
from mongointerface import MongoInterface


class ColorCodes(object):
    def __init__(self,
                p_succ='#01456a',
                fp_succ='#00706e',
                p_fail='#c62200',
                fp_fail='#fe9b00'):
        self.p_succ = mcolors.hex2color(p_succ)  # was success, labeled success
        self.fp_succ = mcolors.hex2color(fp_succ)  # was fail, labeled success
        self.p_fail = mcolors.hex2color(p_fail)  # was fail, labeled fail
        self.fp_fail = mcolors.hex2color(fp_fail)  # was success, labeled fail

    def get_color(self, success, label):
        color = self.p_succ
        if not success and label:
            color = self.fp_succ
        elif not success and not label:
            color = self.p_fail
        elif success and not label:
            color = self.fp_fail
        return color

    def get_pos_color(self, success):
        if success:
            return self.p_succ
        else:
            return self.p_fail

def sample_df(experiment_name, sample_id):
    mf = MongoInterface()
    df = mf.sample_to_df(experiment_name, sample_id)
    df.set_index(['id'], inplace=True)
    return df

def experiment_df(name):
    mf = MongoInterface()
    return mf.experiment_to_df(name)

def plot_makespans(sample_df):
    sim_plt = mpl.figure(num=1, figsize=(20, 10))

    ax = mpl.subplot(111)
    ax.set_title('Tasks Makespan')
    ax.set_xlabel('t')
    ax.set_ylabel('task id')

    start_ts=0
    end_ts=sample_df['end_date'].max(axis=0)

    task_count=sample_df.shape[0]

    ax.set_ylim(0, task_count - 1)
    ax.set_xlim(start_ts, end_ts)

    task_labels=[]
    lines=[]
    colors=[]
    y=np.zeros(2)

    clr_codes=ColorCodes()
    for i in sample_df.index:
        task_labels.append('T {}'.format(i))
        task=sample_df.loc[i]
        sections=list(zip(task['activation_ts'], task['interrupt_ts']))
        # sections = list(zip(tasks.loc[i, 'activation_ts'], tasks.loc[i, 'interrupt_ts']))
        lines.extend(
            [np.concatenate((np.array(x), y)).reshape((2, 2)).T for x in sections])
        colors.extend([clr_codes.get_pos_color(
            task['success'])] * len(sections))
        y += 1

    ax.add_collection(LineCollection(lines, colors=colors))

    ylim=ax.yaxis.get_view_interval()
    def task_format(tick, pos):
        i=int(tick)
        if i >= 0 and i < len(task_labels):
            return task_labels[i]
            # return labels[i]
            # return i
        # if i >= ylim[0] and i <= ylim[1]:
            # return labels[i]
            # return int(tick)
        else:
            return ''

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(task_format))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    mpl.show()

def plot_hist(experiment_df, column):
    pass

def plot_bar(df):
    pass

def avg(df):
    return df.mean()

def avg_exp(df):
    series =  {i:avg(df.loc[i]) for i in df.index.levels[0]}
    avg_df = pd.DataFrame(series).T
    return avg_df

def max_freq(x):
    return x.value_counts().index[0]

def agg_experiment_df(df):
    df = df.groupby('id').agg(
        {'quota': {'max_freq quota': max_freq},
        'pkg': {'max freq pkg': max_freq},
        'priority': {'avg priority': 'mean'},
        'period': {'avg period': 'mean'},
        'numberofjobs': {'avg numberofjobs':'mean'},
        'offset': {'avg offset':'mean'},
        'executiontime': {'avg executiontime':'mean'},
        'criticaltime': {'avg criticaltime':'mean'},
        'deadline':{'avg deadline':'count'},
        'success':{'length':'count'}
        }
    )
    df.columns = df.columns.droplevel()
    return df




if __name__ == '__main__':
    df=sample_df('test', 1)
    plot_makespans(df)