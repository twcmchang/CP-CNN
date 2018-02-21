# $1: GPU ID
# $2: multiplier for sparsity regularization
# $3: multiplier for monotonicity-induced penalty
# $4: multiplier for weight decay
# $5: keep probability for Dropout
# $6: storage directory
# $7: pre-trained model
# ------------------------------------------------------
# First, train with sparsity and monotonicity constraints, prune network, and save a sparse parameter dictionary in save_dir
# Second, aggregate losses at different fidelity levels and then fine-tune the pruned model
# Third, evaluation the performance of different fidelity levels
CUDA_VISIBLE_DEVICES=$1 python3 train.py --prof_type linear --lambda_s $2 --lambda_m $3 --decay $4 --keep_prob $5 --save_dir $6 --init_from $7
CUDA_VISIBLE_DEVICES=$1 python3 train.py --tesla 1 --decay $4 --keep_prob $5 --init_from $6/sparse_dict.npy --save_dir tesla_$6
CUDA_VISIBLE_DEVICES=$1 python3 test.py --keep_prob $5 --init_from tesla_$6/finetune_dict.npy --output tesla_$6/output.csv
