# %load train.py
import os
import time
import argparse
import numpy as np
import tensorflow as tf

from progress.bar import Bar
from ipywidgets import IntProgress
from IPython.display import display
from imgaug import augmenters as iaa

from vgg16_variational_dp import VGG16
from utils import CIFAR10, CIFAR100, gamma_sparsify_VGG16

transform = iaa.Sometimes(
    0.5,
    iaa.Affine(
        scale={"x":(0.9,1.1), "y":(0.9,1.1)},
        translate_percent={"x":(-0.1,0.1), "y":(-0.1,0.1)},
        rotate=(-30,30),
    )
)

arr_spareness = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_from', type=str, default='vgg16.npy', help='pre-trained weights')
    parser.add_argument('--save_dir', type=str, default='save', help='directory to store checkpointed models')
    parser.add_argument('--dataset', type=str, default='CIFAR-10', help='dataset in use')
    parser.add_argument('--prof_type', type=str, default='all-one', help='type of profile coefficient')
    parser.add_argument('--tesla', type=int, default=0, help='task-wise early stopping and loss aggregation')
    parser.add_argument('--atp', type=int, default=0, help='alternative training procedure')
    parser.add_argument('--l1', type=float, default=0.0, help='alternative training procedure')
    parser.add_argument('--l1_diff', type=float, default=0.0, help='alternative training procedure')
    parser.add_argument('--log_dir', type=str, default='log', help='directory containing log text')
    parser.add_argument('--decay', type=float, default=0.0, help='l2 loss of weight')
    parser.add_argument('--keep_prob', type=float, default=1.0, help='dropout keep probability for fc layer')    
    parser.add_argument('--note', type=str, default='', help='argument for taking notes')

    FLAG = parser.parse_args()
    if FLAG.tesla == 0:
        print("===== train 'var_dp' =====")
        train(FLAG)
    else:
        print("===== train TESLA =====")
        train_loss_agg(FLAG)

def train(FLAG):
    print("Reading dataset...")
    if FLAG.dataset == 'CIFAR-10':
        train_data = CIFAR10(train=True)
        test_data  = CIFAR10(train=False)
        vgg16 = VGG16(classes=10)
    elif FLAG.dataset == 'CIFAR-100':
        train_data = CIFAR100(train=True)
        test_data  = CIFAR100(train=False)
        vgg16 = VGG16(classes=100)
    else:
        raise ValueError("dataset should be either CIFAR-10 or CIFAR-100.")
    print("Build VGG16 models for %s..."% FLAG.dataset)

    Xtrain, Ytrain = train_data.train_data, train_data.train_labels
    Xtest, Ytest = test_data.test_data, test_data.test_labels
  
    vgg16.build(vgg16_npy_path=FLAG.init_from, prof_type=FLAG.prof_type, conv_pre_training=True, fc_pre_training=False)
    vgg16.sparsity_train(l1_gamma=FLAG.l1, l1_gamma_diff=FLAG.l1_diff, decay=FLAG.decay, keep_prob=FLAG.keep_prob)

    # build model using  dp
    # dp = [(i+1)*0.05 for i in range(1,20)]
    # vgg16.set_idp_operation(dp=dp)

    # define tasks
    tasks = ['var_dp']
    print(tasks)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=len(tasks))
    
    checkpoint_path = os.path.join(FLAG.save_dir, 'model.ckpt')
    tvars_trainable = tf.trainable_variables()
    
    #for rm in vgg16.gamma_var:
    #    tvars_trainable.remove(rm)
    #    print('%s is not trainable.'% rm)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # hyper parameters
        learning_rate =  5e-4 #adam
        batch_size = 64
        epoch = 50
        # alpha = 0.5
        early_stop_patience = 4
        min_delta = 0.0001

        # recorder
        epoch_counter = 0

        # optimizer
        # opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # tensorboard writer
        # writer = tf.summary.FileWriter(FLAG.log_dir, sess.graph)

        # progress bar
        ptrain = IntProgress()
        pval = IntProgress()
        display(ptrain)
        display(pval)
        ptrain.max = int(Xtrain.shape[0]/batch_size)
        pval.max = int(Xtest.shape[0]/batch_size)
        
        # initial task
        cur_task = tasks[0]
        obj = vgg16.loss_dict[tasks[0]]

        # optimizer
        train_op = opt.minimize(obj, var_list=tvars_trainable)
        spareness = vgg16.spareness(thresh=0.05)
        arr_spareness.append(sess.run(spareness))
        print("initial spareness: %s" % sess.run(spareness))
        # re-initialize
        initialize_uninitialized(sess)

        # reset due to adding a new task
        patience_counter = 0
        current_best_val_loss = 100000 # a large number

        # optimize when the aggregated obj
        while(patience_counter < early_stop_patience):
            stime = time.time()
            bar_train = Bar('Training', max=int(Xtrain.shape[0]/batch_size), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
            bar_val =  Bar('Validation', max=int(Xtest.shape[0]/batch_size), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
            
            # training an epoch
            for i in range(int(Xtrain.shape[0]/batch_size)):
                st = i*batch_size
                ed = (i+1)*batch_size

                # image augment
                augX = transform.augment_images(Xtrain[st:ed,:,:,:])
                
                # sess.run([train_op], feed_dict={vgg16.x: Xtrain[st:ed,:,:,:],
                #                                 vgg16.y: Ytrain[st:ed,:]})
                sess.run([train_op], feed_dict={vgg16.x: augX,
                                                vgg16.y: Ytrain[st:ed,:],
                                                vgg16.is_train: True})
                ptrain.value +=1
                ptrain.description = "Training %s/%s" % (i, ptrain.max)
                bar_train.next()

            # validation
            val_loss = 0
            val_accu = 0
            for i in range(int(Xtest.shape[0]/200)):
                st = i*200
                ed = (i+1)*200
                loss, accu = sess.run([obj, vgg16.accu_dict[cur_task]],
                                    feed_dict={vgg16.x: Xtest[st:ed,:],
                                                vgg16.y: Ytest[st:ed,:],
                                                vgg16.is_train: False})
                val_loss += loss
                val_accu += accu
                pval.value += 1
                pval.description = "Testing %s/%s" % (i, pval.value)
            val_loss = val_loss/pval.value
            val_accu = val_accu/pval.value

            arr_spareness.append(sess.run(spareness)) 
            print("\nspareness: %s" % sess.run(spareness))
            # early stopping check
            if (current_best_val_loss - val_loss) > min_delta:
                current_best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # shuffle Xtrain and Ytrain in the next epoch
            idx = np.random.permutation(Xtrain.shape[0])
            Xtrain, Ytrain = Xtrain[idx,:,:,:], Ytrain[idx,:]

            # epoch end
            # writer.add_summary(epoch_summary, epoch_counter)
            epoch_counter += 1

            ptrain.value = 0
            pval.value = 0
            bar_train.finish()
            bar_val.finish()

            print("Epoch %s (%s), %s sec >> obj loss: %.4f, task at %s: %.4f" % (epoch_counter, patience_counter, round(time.time()-stime,2), val_loss, cur_task, val_accu))
        saver.save(sess, checkpoint_path, global_step=epoch_counter)

        para_dict = sess.run(vgg16.para_dict)
        np.save(os.path.join(FLAG.save_dir, "para_dict.npy"), para_dict)
        print("save in %s" % os.path.join(FLAG.save_dir, "para_dict.npy"))

        sp, rcut = gamma_sparsify_VGG16(para_dict, thresh=0.05)
        np.save(os.path.join(FLAG.save_dir,"sparse_dict.npy"), sp)
        print("sparsify %s in %s" % (np.round(1-rcut,3), os.path.join(FLAG.save_dir, "sparse_dict.npy")))
        #writer.close()
        arr_spareness.append(1-rcut)
        np.save(os.path.join(FLAG.save_dir,"sprocess.npy"), arr_spareness)
def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v,f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars): 
            sess.run(tf.variables_initializer(not_initialized_vars))

def train_loss_agg(FLAG):
    print("Reading dataset...")
    if FLAG.dataset == 'CIFAR-10':
        train_data = CIFAR10(train=True)
        test_data  = CIFAR10(train=False)
        tasks = ['100', '75', '50', '25', '15']
        vgg16 = VGG16(classes=10)
    elif FLAG.dataset == 'CIFAR-100':
        train_data = CIFAR100(train=True)
        test_data  = CIFAR100(train=False)
        tasks = ['100', '75', '50', '25']
        vgg16 = VGG16(classes=100)
    else:
        raise ValueError("dataset should be either CIFAR-10 or CIFAR-100.")
    print("Build VGG16 models for %s..."% FLAG.dataset)

    Xtrain, Ytrain = train_data.train_data, train_data.train_labels
    Xtest, Ytest = test_data.test_data, test_data.test_labels
  
    vgg16.build(vgg16_npy_path=FLAG.init_from, prof_type=FLAG.prof_type, conv_pre_training=True, fc_pre_training=True)

    # build model using  dp
    dp = [(i+1)*0.05 for i in range(1,20)]
    vgg16.set_idp_operation(dp=dp, decay=FLAG.decay ,keep_prob=FLAG.keep_prob)

    # define tasks
    print(tasks)

    # loss aggregation
    obj = 0.0
    for cur_task in tasks:
        obj += vgg16.loss_dict[cur_task]
    
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=len(tasks))
    checkpoint_path = os.path.join(FLAG.save_dir, 'model.ckpt')

    tvars_trainable = tf.trainable_variables()
    for rm in vgg16.gamma_var:
       tvars_trainable.remove(rm)
       print('%s is not trainable.'% rm)

    for var in tvars_trainable:
        if '_bn_' in var.name:
            tvars_trainable.remove(var)
            print('%s is not trainable.'% var)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

       # hyper parameters
        learning_rate = 5e-4 # adam # 4e-3 #sgd
        batch_size = 64
        epoch = 50
        # alpha = 0.5
        early_stop_patience = 4
        min_delta = 0.0001

        # recorder
        epoch_counter = 0

        # optimizer
        # opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # optimizer
        train_op = opt.minimize(obj, var_list=tvars_trainable)

        # progress bar
        ptrain = IntProgress()
        pval = IntProgress()
        display(ptrain)
        display(pval)
        ptrain.max = int(Xtrain.shape[0]/batch_size)
        pval.max = int(Xtest.shape[0]/batch_size)
        
        spareness = vgg16.spareness(thresh=0.05)
        print("initial spareness: %s" % sess.run(spareness))
        # re-initialize
        initialize_uninitialized(sess)

        # reset due to adding a new task
        patience_counter = 0
        current_best_val_loss = 100000 # a large number

        # optimize when the aggregated obj
        while(patience_counter < early_stop_patience):
            stime = time.time()
            bar_train = Bar('Training', max=int(Xtrain.shape[0]/batch_size), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
            bar_val =  Bar('Validation', max=int(Xtest.shape[0]/batch_size), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')

            # training an epoch
            for i in range(int(Xtrain.shape[0]/batch_size)):
                st = i*batch_size
                ed = (i+1)*batch_size

                augX = transform.augment_images(Xtrain[st:ed,:,:,:])

                sess.run([train_op], feed_dict={vgg16.x: augX,
                                                vgg16.y: Ytrain[st:ed,:],
                                                vgg16.is_train: False})
                ptrain.value +=1
                ptrain.description = "Training %s/%s" % (i, ptrain.max)
                bar_train.next()

            # validation
            val_loss = 0
            val_accu = 0
            for i in range(int(Xtest.shape[0]/200)):
                st = i*200
                ed = (i+1)*200
                loss, accu = sess.run([obj, vgg16.accu_dict[cur_task]],
                                    feed_dict={vgg16.x: Xtest[st:ed,:],
                                                vgg16.y: Ytest[st:ed,:],
                                                vgg16.is_train: False})
                val_loss += loss
                val_accu += accu
                pval.value += 1
                pval.description = "Testing %s/%s" % (i, pval.value)
            val_loss = val_loss/pval.value
            val_accu = val_accu/pval.value

            print("\nspareness: %s" % sess.run(spareness))
            # early stopping check
            if (current_best_val_loss - val_loss) > min_delta:
                current_best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # shuffle Xtrain and Ytrain in the next epoch
            idx = np.random.permutation(Xtrain.shape[0])
            Xtrain, Ytrain = Xtrain[idx,:,:,:], Ytrain[idx,:]

            # epoch end
            # writer.add_summary(epoch_summary, epoch_counter)
            epoch_counter += 1

            ptrain.value = 0
            pval.value = 0
            bar_train.finish()
            bar_val.finish()

            print("Epoch %s (%s), %s sec >> obj loss: %.4f, task at %s: %.4f" % (epoch_counter, patience_counter, round(time.time()-stime,2), val_loss, cur_task, val_accu))
        saver.save(sess, checkpoint_path, global_step=epoch_counter)

        para_dict = sess.run(vgg16.para_dict)
        np.save(os.path.join(FLAG.save_dir, "para_dict.npy"), para_dict)
if __name__ == '__main__':
    main()
