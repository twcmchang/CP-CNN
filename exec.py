import os

l1 = [0.0002]
l1_diff = [0.0002]
count = 3
for p in l1:
	for a in l1_diff:
		print("test:{0}, l1:{1}, l1_diff:{2}".format(count,p,a))
		os.system("CUDA_VISIBLE_DEVICES=6 python3 train.py --l1 {0} --l1_diff {1} --save_dir save_grad_{2} --prof_type linear".format(p,a,count))
		count +=1
