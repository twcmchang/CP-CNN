#!/bin/bash
# $1: GPU ID
# $2: keep probability for Dropout
# $3: testing model
# $4: fidelity level under test
# $5: output filename
CUDA_VISIBLE_DEVICES=$1 python3 test.py --keep_prob $2 --init_from $3 --fidelity $4 --output $5