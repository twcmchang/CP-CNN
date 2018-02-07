import os
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from progress.bar import Bar
from ipywidgets import IntProgress
from IPython.display import display
from vgg16_variational_dp import VGG16
from utils import CIFAR10, CIFAR100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_from', type=str, default='vgg16.npy', help='pre-trained weights')
    parser.add_argument('--save_dir', type=str, default=None, help='directory to store checkpointed models')
    parser.add_argument('--dataset', type=str, default='CIFAR-10', help='dataset in use')
    parser.add_argument('--prof_type', type=str, default='all-one', help='type of profile coefficient')
    parser.add_argument('--output', type=str, default='output.csv', help='output filename (csv)')
    parser.add_argument('--atp', type=int, default=0, help='alternative training procedure')
    parser.add_argument('--keep_prob', type=float, default=1.0, help='dropout keep probability for fc layer') 
    # parser.add_argument('--log_dir', type=str, default='log', help='directory containing log text')
    # parser.add_argument('--note', type=str, default='', help='argument for taking notes')

    FLAG = parser.parse_args()
    test(FLAG)

def test(FLAG):
    print("Reading dataset...")
    if FLAG.dataset == 'CIFAR-10':
        test_data  = CIFAR10(train=False)
        vgg16 = VGG16(classes=10)
    elif FLAG.dataset == 'CIFAR-100':
        test_data  = CIFAR100(train=False)
        vgg16 = VGG16(classes=100)
    else:
        raise ValueError("dataset should be either CIFAR-10 or CIFAR-100.")

    Xtest, Ytest = test_data.test_data, test_data.test_labels

    print("Build VGG16 models...")
    vgg16.build(vgg16_npy_path=FLAG.init_from, prof_type=FLAG.prof_type, conv_pre_training=True, fc_pre_training=True)

    # build model using  dp
    dp = [(i+1)*0.05 for i in range(1,20)]
    vgg16.set_idp_operation(dp=dp, keep_prob=FLAG.keep_prob)

    with tf.Session() as sess:
        if FLAG.save_dir is not None:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(FLAG.save_dir)
            
            if ckpt and ckpt.model_checkpoint_path:
                count = 0
                for checkpoint in ckpt.all_model_checkpoint_paths:
                    saver.restore(sess, checkpoint)
                    print("Model restored %s" % checkpoint)
                    sess.run(tf.global_variables())
                    print("Initialized")
                    count += 1
                    output = []
                    # for dp_i in dp:
                    #     accu = sess.run(vgg16.accu_dict[str(int(dp_i*100))], feed_dict={vgg16.x: Xtest[:5000,:], vgg16.y: Ytest[:5000,:]})
                    #     accu2 = sess.run(vgg16.accu_dict[str(int(dp_i*100))], feed_dict={vgg16.x: Xtest[5000:,:], vgg16.y: Ytest[5000:,:]})
                    #     output.append((accu+accu2)/2)
                    #     print("At DP={dp:.4f}, accu={perf:.4f}".format(dp=dp_i, perf=(accu+accu2)/2))
                    for dp_i in dp:
                        val_accu = 0
                        n_batch = 0
                        for i in range(int(Xtest.shape[0]/200)):
                            st = i*200
                            ed = (i+1)*200
                            accu = sess.run(vgg16.accu_dict[str(int(dp_i*100))],
                                                feed_dict={vgg16.x: Xtest[st:ed,:],
                                                        vgg16.y: Ytest[st:ed,:],
                                                        vgg16.is_train: False})
                            val_accu += accu
                            n_batch +=1
                        output.append(val_accu/n_batch)
                    res = pd.DataFrame.from_dict({'DP':[int(dp_i*100) for dp_i in dp],'accu':output})
                    res.to_csv("task%s_%s" % (count, FLAG.output), index=False)
                    print("Write into task%s_%s" % (count, FLAG.output))

        else:
            count = 0
            sess.run(tf.global_variables_initializer())
            sess.run(tf.global_variables())
            print("Initialized")
            print(sess.run(vgg16.para_dict['conv1_1_bn_mean']))
            print(sess.run(vgg16.para_dict['conv1_1_bn_variance']))
            count += 1
            output = []
            for dp_i in dp:
                accu = sess.run(vgg16.accu_dict[str(int(dp_i*100))], feed_dict={vgg16.x: Xtest[:5000,:], vgg16.y: Ytest[:5000,:], vgg16.is_train: False})
                accu2 = sess.run(vgg16.accu_dict[str(int(dp_i*100))], feed_dict={vgg16.x: Xtest[5000:,:], vgg16.y: Ytest[5000:,:], vgg16.is_train: False})
                output.append((accu+accu2)/2)
                print("At DP={dp:.4f}, accu={perf:.4f}".format(dp=dp_i, perf=(accu+accu2)/2))
            res = pd.DataFrame.from_dict({'DP':[int(dp_i*100) for dp_i in dp],'accu':output})
            res.to_csv("task%s_%s" % (count, FLAG.output), index=False)
            print("Write into task%s_%s" % (count, FLAG.output))


if __name__ == '__main__':
	main()
