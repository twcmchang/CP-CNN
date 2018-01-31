import os

l1 = [0.002, 0.0002]
l1_diff = [0.02, 0.002, 0.0002]
count = 1
for p in l1:
	for a in l1_diff:
		print("test:{0}, l1:{1}, l1_diff:{2}".format(count,p,a))
		os.system("CUDA_VISIBLE_DEVICES=0 python3 train.py --l1 {0} --l1_diff {1} --save_dir save_linear_{2} --prof_type linear".format(p,a,count))
		count +=1
