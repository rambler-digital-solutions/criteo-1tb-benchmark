#!/usr/bin/env bash


TRAIN="${1}"
TEST="${2}"

if [[ "${TRAIN}" == "" || "${TEST}" == "" ]]; then
    echo "Usage: $0 train test"
    exit 1
fi

TIME="${TRAIN}.time"
MODEL="${TRAIN}.model"
PREDICTIONS="${TEST}.predictions"


VW_OPTS=($(cat vw.conf))
echo VW_OPTS = "${VW_OPTS[@]}"

/usr/local/bin/time -v --output="${TIME}" \
    vw83 --link=logistic --loss_function=logistic -d "${TRAIN}" -f "${MODEL}" "${VW_OPTS[@]}"

vw83 -i "${MODEL}" --loss_function=logistic -t -d "${TEST}" -p "${PREDICTIONS}"


METRICS="metrics.tsv"
if ! [[ -e ${METRICS} ]]; then
    echo -e "Engine\tTrain size\tROC AUC\tLog loss\tTrain time\tMaximum memory\tCPU load" | tee "${METRICS}"
fi

python measure.py "vw" "${TRAIN}" "${TEST}" | tee -a "${METRICS}"

rm "${TIME}" "${PREDICTIONS}"
