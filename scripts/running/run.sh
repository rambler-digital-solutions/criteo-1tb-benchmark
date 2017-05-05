#!/usr/bin/env bash


DATA_PREFIX="./data"

test_vw="${DATA_PREFIX}/data.test.1kk.vw"
test_xgb="${DATA_PREFIX}/data.test.1kk.libsvm"

for train_num in 10k 30k 100k 300k 1kk 3kk 10kk 30kk 100kk 300kk 1kkk 3kkk; do
    echo " * * * Train size ${train_num} lines * * *"

    train_vw="${DATA_PREFIX}/data.train.${train_num}.vw"
    train_xgb="${DATA_PREFIX}/data.train.${train_num}.libsvm"

    echo "Running VW with train ${train_vw} and ${test_vw}"
    ./vw.sh "${train_vw}" "${test_vw}"

    echo "Running XGBoost with train ${train_xgb} and ${test_xgb}"
    ./xgb.sh "${train_xgb}" "${test_xgb}"

    echo "Running XGBoost (out-of-core) with train ${train_xgb} and ${test_xgb}"
    ./xgb.ooc.sh "${train_xgb}" "${test_xgb}"
done
