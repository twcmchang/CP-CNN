from PIL import Image
import os
import errno
import sys
import pickle
import tarfile
import zipfile
import numpy as np
from urllib.request import urlretrieve

class CIFAR10(object):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    root = 'data/'
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    num_classes = 10
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, train=True):
        if not os.path.exists(os.path.join(self.root,self.base_folder)):
            self.maybe_download_and_extract()

        self.train = train  # training set or test set

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.train_labels = self._one_hot_encoded(self.train_labels)
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.test_labels = self._one_hot_encoded(self.test_labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        return fmt_str

    def _one_hot_encoded(self, class_numbers):
        return np.eye(self.num_classes, dtype=float)[class_numbers]

    def _print_download_progress(self,count, block_size, total_size):
        pct_complete = float(count * block_size) / total_size
        msg = "\r- Download progress: {0:.1%}".format(pct_complete)
        sys.stdout.write(msg)
        sys.stdout.flush()

    def maybe_download_and_extract(self):
        main_directory = self.root
        if not os.path.exists(main_directory):
            os.makedirs(main_directory)

        filename = self.url.split('/')[-1]
        file_path = os.path.join(main_directory, filename)
        zip_cifar = file_path
        if not os.path.exists(file_path):
            file_path, _ = urlretrieve(url=self.url, filename=file_path, reporthook=self._print_download_progress)

        print()
        print("Download finished. Extracting files.")
        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(main_directory)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(main_directory)
        print("Done.")

        os.remove(zip_cifar)


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    root = 'data/'
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    num_classes = 100
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]


def gammaSparsifyVGG16(para_dict, thresh=0.5):
    last = None
    sparse_dict = {}
    N_total, N_remain = 0., 0.
    for k, v in sorted(para_dict.items()):
        if 'gamma' in k:
            # trim networks based on gamma
            gamma = v                      
            this = np.where(np.abs(gamma) > thresh)[0]
            sparse_dict[k] = gamma[this] 
            
            # get the layer name
            key = str.split(k,'_gamma')[0]
            
            # trim conv
            conv_, bias_ = para_dict[key]
            conv_ = conv_[:,:,:,this]
            if last is not None:
                conv_ = conv_[:,:,last,:]
            bias_ = bias_[this]
            sparse_dict[key] = [conv_, bias_]
            
            # get corresponding beta, bn_mean, bn_variance
            sparse_dict[key+"_beta"] = para_dict[key+"_beta"][this]
            sparse_dict[key+"_bn_mean"] = para_dict[key+"_bn_mean"][this]
            sparse_dict[key+"_bn_variance"] = para_dict[key+"_bn_variance"][this]
            
            # update
            last = this
            print('%s from %s to %s : %s ' % (k, len(gamma), len(this), len(this)/len(gamma)))
            N_total += len(gamma)
            N_remain += len(this)
    print('sparsify %s percentage' % (N_remain/N_total))
    W_, b_ = para_dict['fc_1']
    W_ = W_[last,:]
    sparse_dict['fc_1'] = [W_, b_]
    sparse_dict['fc_2'] = para_dict['fc_2']
    return sparse_dict, N_remain/N_total

def dpSparsifyVGG16(para_dict, dp):
    """
    dp: usage percentage of channels in each layer
    """
    new_dict = {}
    first = True
    for k,v in sorted(para_dict.items()):
        if 'conv1_1_' in k:
            new_dict[k] = v
        elif 'bn_mean' in k:
            new_dict[k] = v[:int(len(v)*dp)]
        elif 'bn_variance' in k:
            new_dict[k] = v[:int(len(v)*dp)]
        elif 'gamma' in k:
            new_dict[k] = v[:int(len(v)*dp)]
        elif 'beta' in k:
            new_dict[k] = v[:int(len(v)*dp)]
        elif 'fc_1' in k:
            O = v[0].shape[0]
            new_dict[k] = [v[0][:int(O*dp),:], v[1]]
        elif 'conv' in k:
            O = v[0].shape[3]
            if first:
                new_dict[k] = v[0][:,:,:,:], v[1][:] #int(O*dp)
                first = False
                last = O
            else:
                new_dict[k] = v[0][:,:,:last,:int(O*dp)], v[1][:int(O*dp)]
                last = int(O*dp)
        else:
            new_dict[k] = v
            continue
    return new_dict

def count_number_params(para_dict):
    n = 0
    for k,v in sorted(para_dict.items()):
        if 'bn_mean' in k:
            continue
        elif 'bn_variance' in k:
            continue
        elif 'gamma' in k:
            continue
        elif 'beta' in k:
            continue
        elif 'conv' in k or 'fc' in k:
            n += get_params_shape(v[0].shape.as_list())
            n += get_params_shape(v[1].shape.as_list())
    return n

def get_params_shape(shape):
    n = 1
    for dim in shape:
        n = n*dim
    return n

def count_flops(para_dict, net_shape):
    input_shape = (3 ,32 ,32) # Format:(channels, rows,cols)
    total_flops_per_layer = 0
    input_count = 0
    for k,v in sorted(para_dict.items()):
        if 'bn_mean' in k:
            continue
        elif 'bn_variance' in k:
            continue
        elif 'gamma' in k:
            continue
        elif 'beta' in k:
            continue
        elif 'fc' in k:
            continue
        elif 'conv' in k:
            conv_filter = v[0].shape.as_list()[3::-1] # (64 ,3 ,3 ,3)  # Format: (num_filters, channels, rows, cols)
            stride = 1
            padding = 1

            if conv_filter[1] == 0:
                n = conv_filter[2] * conv_filter[3] # vector_length
            else:
                n = conv_filter[1] * conv_filter[2] * conv_filter[3]  # vector_length

            flops_per_instance = n + ( n -1)    # general defination for number of flops (n: multiplications and n-1: additions)

            num_instances_per_filter = (( input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1  # for rows
            num_instances_per_filter *= ((input_shape[1] - conv_filter[2] + 2 * padding) / stride) + 1  # multiplying with cols

            flops_per_filter = num_instances_per_filter * flops_per_instance
            total_flops_per_layer += flops_per_filter * conv_filter[0]  # multiply with number of filters

            total_flops_per_layer += conv_filter[0] * input_shape[1] * input_shape[2]

            input_shape = net_shape[input_count].as_list()[3:0:-1]
            input_count +=1

    total_flops_per_layer += net_shape[-1].as_list()[3] * 512 *2 + 512*10*2
    return total_flops_per_layer

def countFlopsParas(net):
    total_flops = count_flops(net.para_dict, net.net_shape)
    if total_flops / 1e9 > 1:   # for Giga Flops
        print(total_flops/ 1e9 ,'{}'.format('GFlops'))
    else:
        print(total_flops / 1e6 ,'{}'.format('MFlops'))

    total_params = count_number_params(net.para_dict)

    if total_params / 1e9 > 1:   # for Giga Flops
        print(total_params/ 1e9 ,'{}'.format('G'))
    else:
        print(total_params / 1e6 ,'{}'.format('M'))
    
    return total_flops, total_params