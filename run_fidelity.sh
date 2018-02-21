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
CUDA_VISIBLE_DEVICES=$1 python3 test.py --keep_prob $2 --init_from $3 --fidelity $4 --output $5