from __future__ import print_function

import argparse

import cycler
import pandas

from matplotlib import pyplot


def extract_data_for_plotting(df, what):
    return reduce(
        lambda left, right: pandas.merge(
            left,
            right,
            how='outer',
            on='Train size',
        ),
        map(
            lambda name: (
                df[df.Engine == name][['Train size', what]]
                .rename(columns={what: name})
            ),
            df.Engine.unique(),
        ),
    )


def plot_stuff(df, what, ylabel=None, **kwargs):
    data = extract_data_for_plotting(df, what).set_index('Train size')
    ax = data.plot(
        figsize=(6, 6),
        title=what,
        grid=True,
        linewidth=2.0,
        marker='o',
        **kwargs
    )
    ax.legend(loc='best')
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.grid(which='major', linestyle='-')
    ax.grid(which='minor', linestyle=':')

    if experiment_name is not None:
        ax.get_figure().savefig(what + '.' + experiment_name + '.png')
    else:
        ax.get_figure().savefig(what + '.png')


parser = argparse.ArgumentParser()

parser.add_argument('-i', '--input', type=str, default='metrics.tsv',
                    help='input file')

parser.add_argument('-n', '--name', type=str,
                    help='experiment name')

parser.add_argument('-p', '--perf', action='store_true',
                    help='build perf graphs')

parser.add_argument('-c', '--colors', type=str,
                    help='color cycle')

parser.add_argument('-l', '--logloss',
                    help='log loss scale')

args = parser.parse_args()


metrics_file = args.input
experiment_name = args.name
perf_graphs = args.perf

if args.colors is not None:
    color_cycle = args.colors.split(',')
    pyplot.rc('axes', prop_cycle=(cycler.cycler('color', color_cycle)))

df = (
    pandas
    .read_csv(metrics_file, sep='\t')
    .sort_values(by=['Engine', 'Train size'])
)

plot_stuff(df, 'ROC AUC', logx=True)

if args.logloss is not None:
    ll_from, ll_to = map(float, args.logloss.split(','))
    plot_stuff(df, 'Log loss', logx=True, ylim=(ll_from, ll_to))
else:
    plot_stuff(df, 'Log loss', logx=True)

plot_stuff(df, 'Train time', loglog=True, ylabel='s')

if perf_graphs:
    plot_stuff(df, 'Maximum memory', loglog=True, ylabel='bytes')
    plot_stuff(df, 'CPU load', logx=True)
