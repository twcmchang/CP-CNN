# Channel-Prioritized Convolutional Neural Network with Sparsity and Multi-fidelity
This repository contains the code to reproduce the core results from the paper [Channel-Prioritized Convolutional Neural Networks with Sparsity and Multi-fidelity](https://openreview.net/pdf?id=S1qru_kDf) in the review process.

# Dependencies
This work uses Python 3.6.0. Before running the code, you have to install
- tensorflow==1.4.0
- numpy==1.13.0
- pandas==0.20.3
- Pillow==4.3.0
- progress==1.3

The above dependencies can be installed using pip by running
```
pip install -r requirement.txt
```

# Usage
To have a quick start on the experiment of CIFAR-10 by running
```
bash quick_start.sh <GPU_ID>
```

Training stage for channel prioritization and network sparsity
```
python3 train.py  --init_from <pre-trained_net_params.npy> --save_dir <save_directory> --tesla 0 --keep_prob 1  --lambda_s 0.001 --lambda_m 0.001 --decay 0.00005 --prof_type linear
```

Fine-tuning stage (set tesla=1) for loss aggregation
```
python3 train.py --init_from <pruned_net_params.npy> --save_dir <save_directory> --tesla 1 --keep_prob 1  --prof_type linear
```

Testing multi-fidelity inference
```
python3 test.py --init_from <finetuned_net_params.npy> --output <output_file> --keep_prob 1
```

To run the experiment on CIFAR-100, please add another arguement ```--dataset CIFAR-100``` in above commands.


If you have questions, please write an email to cmchang@iis.sinica.edu.tw.

