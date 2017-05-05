import sys
from sklearn.metrics import (
    auc,
    log_loss,
    roc_curve,
)


engine = sys.argv[1]
train_file = sys.argv[2]
test_file = sys.argv[3]

scores_file = test_file + '.predictions'
time_file = train_file + '.time'


def get_last_in_line(s):
    return s.rstrip().split( )[-1]

def parse_elapsed_time(s):
    return reduce(lambda a, b: a * 60 + b, map(float, get_last_in_line(s).split(':')))

def parse_max_memory(s):
    return int(get_last_in_line(s)) * 1024

def parse_cpu(s):
    return float(get_last_in_line(s).rstrip('%')) / 100


elapsed = -1
memory = -1
cpu = -1

with open(time_file, 'rb') as f:
    for line in f:
        if 'Elapsed (wall clock) time' in line:
            elapsed = parse_elapsed_time(line)
        elif 'Maximum resident set size' in line:
            memory = parse_max_memory(line)
        elif 'Percent of CPU' in line:
            cpu = parse_cpu(line)


with open(test_file, 'rb') as f:
    labels = [line.rstrip().split(' ')[0] == '1' for line in f]

with open(scores_file, 'rb') as f:
    scores = [float(line.rstrip().split(' ')[0]) for line in f]

fpr, tpr, _ = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)
ll = log_loss(labels, scores)


try:
    train_size = int(train_file.split('/')[-1].split('.')[2].replace('k', '000'))
except:
    train_size = 0

print '\t'.join(map(str, [engine, train_size, roc_auc, ll, elapsed, memory, cpu]))
