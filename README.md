# FPDS

- Sparsity by channel-prioritized training
```
python3 train.py --init_from vgg16.npy --tesla 0 --keep_prob 1 --prof_type linear --l1 0.0005 --l1_diff 0.001 --decay 0.00005 --save_dir save_dir
```

- Multi-fidelity by TESLA training
```
python3 train.py --init_from save_dir/sparse_dict.npy --tesla 1 --keep_prob 1 --prof_type linear --save_dir save_tesla_v1
```


