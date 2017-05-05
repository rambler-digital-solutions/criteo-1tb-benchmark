#!/usr/bin/env bash

python plots.py -i metrics.old.tsv -n 'why_optimize' -c 'b,k,y'
python plots.py -i metrics.selection.tsv -n 'cluster_selection' -c 'k,r,g,b'
python plots.py -i vw_xgb.tsv -n 'local' -p -c 'b,k,y'
python plots.py -i metrics.lr_hash_size.tsv -n 'lr_hash_size' -c 'r,g,b,k,y'
python plots.py -i metrics.cluster.tsv -n 'cluster' -c 'r,g,b,k,y'
python plots.py -i metrics.tsv -n 'local_and_cluster' -c 'r,g,b,k,y' -l '0.13,0.15'
