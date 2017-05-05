#!/usr/bin/env bash


TRAIN="${1}"
TEST="${2}"

TRAIN_OOC="${TRAIN}#${TRAIN}.cache"

if [[ "${TRAIN}" == "" || "${TEST}" == "" ]]; then
    echo "Usage: $0 train test"
    exit 1
fi

TIME="${TRAIN}.time"
MODEL="${TRAIN}.ooc.model"
PREDICTIONS="${TEST}.predictions"


/usr/local/bin/time -v --output="${TIME}" \
    xgboost xgb.conf data="${TRAIN_OOC}" model_out="${MODEL}"

xgboost xgb.conf task=pred test:data="${TEST}" model_in="${MODEL}" name_pred="${PREDICTIONS}"


METRICS="metrics.tsv"
if ! [[ -e ${METRICS} ]]; then
    echo -e "Engine\tTrain size\tROC AUC\tLog loss\tTrain time\tMaximum memory\tCPU load" | tee "${METRICS}"
fi

python measure.py "xgb.ooc" "${TRAIN}" "${TEST}" | tee -a "${METRICS}"

rm "${TIME}" "${PREDICTIONS}" "${TRAIN}.cache"* "${TEST}.buffer"
