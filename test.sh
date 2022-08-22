#!/bin/bash

TEST_ROOT=$1
TEST_ITER=${2:-4000}
GPU_ID=${3:-0}
CONFIG_FILE="${TEST_ROOT}/${TEST_ROOT:26}.py"
CHECKPOINT_FILE="${TEST_ROOT}/iter_${TEST_ITER}.pth"
SHOW_DIR="${TEST_ROOT}/preds/"
echo 'Config File:' $CONFIG_FILE
echo 'Checkpoint File:' $CHECKPOINT_FILE
echo 'Predictions Output Directory:' $SHOW_DIR
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --eval mIoU mFscore --show-dir ${SHOW_DIR} --opacity 1 --gpu-id ${GPU_ID}
