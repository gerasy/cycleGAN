import os
import urllib
import zipfile

from os import listdir
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np

import random
import time
import sys
import datetime
import pickle

import itertools
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image

import os
import numpy as np
import imageio
import gzip
from zipfile import ZipFile

from mlxtend.data import loadlocal_mnist

try:
    from urllib.error import URLError
    from urllib.request import urlretrieve
except ImportError:
    from urllib2 import URLError
    from urllib import urlretrieve

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    Saliency
)




def get_data_MNIST(version, n_train_samples, n_test_samples):
    
    resources = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
    ]
    
    # DOWNLOAD
    # mnist
    mnist_url = 'http://yann.lecun.com/exdb/mnist/'
    mnist_path = './data/mnist/'

    if not os.path.exists(mnist_path):
        print('create mnist folder')
        os.makedirs(mnist_path)
    
    for r in resources:
        r_path = mnist_path + r
        if not os.path.exists(r_path):
            download_path = mnist_url + r
            save_path = 'data/mnist/' + r
            #print(download_path)
            #print(save_path)
            print('Downloading {}'.format(r))
            try: 
                urlretrieve(download_path, save_path)
            except URLError:
                raise RuntimeError('Error downloading resource!')

    # mnist c
    mnistc_url = 'https://zenodo.org/record/3239543/files/mnist_c.zip'
    mnistc_path = './data/mnistc/'
    mnistc_zip_path = mnistc_path + 'mnist_c.zip'

    if not os.path.exists(mnistc_path):
        print('create mnistc folder')
        os.makedirs(mnistc_path)

    if not os.path.exists(mnistc_zip_path):
        print('Downloading mnist_c')
        try: 
            urlretrieve(mnistc_url, mnistc_zip_path)
        except URLError:
            raise RuntimeError('Error downloading resource!')
    

    # UNZIP
    # mnist
    for r in resources:
        zipped_path = mnist_path + r
        unzipped_path = os.path.splitext(zipped_path)[0]

        if not os.path.exists(unzipped_path):
            print('Unzipping {}'.format(r))
            with gzip.open(zipped_path,  'rb') as zipped_file:
                with open(unzipped_path, 'wb') as unzipped_file:
                    unzipped_file.write(zipped_file.read())

    # mnistc
    zipped_path = mnistc_zip_path
    unzipped_path = mnistc_path + 'mnist_c/' + version

    if not os.path.exists(unzipped_path):
        print('unzipping mnistc')
        zip_file = ZipFile(zipped_path)
        for f in zip_file.namelist():
            print(f)
            if f.startswith('mnist_c/'+version):
                zip_file.extract(f, mnistc_path)

    # CREATE FOLDERS
    path = './data/mnist2{}_{}'.format(version, n_train_samples)
    
    if not os.path.exists(path):
        print('create data folder')
        os.makedirs(path)

    A_path_train = path + '/trainA/'
    A_path_test = path + '/testA/'

    if not os.path.exists(A_path_train):
        print('create trainA')
        os.makedirs(A_path_train)

    if not os.path.exists(A_path_test):
        print('create testA')
        os.makedirs(A_path_test)

    B_path_train = path + '/trainB/'
    B_path_test = path + '/testB/'

    if not os.path.exists(B_path_train):
        print('create trainB')
        os.makedirs(B_path_train)

    if not os.path.exists(B_path_test):
        print('create testB')
        os.makedirs(B_path_test)


    # SAVE DATA
    # mnist
    X_train, y_train = loadlocal_mnist(
        images_path = mnist_path + 'train-images-idx3-ubyte',
        labels_path = mnist_path + 'train-labels-idx1-ubyte'
    )
    
    X_test, y_test = loadlocal_mnist(
        images_path = mnist_path + 't10k-images-idx3-ubyte',
        labels_path = mnist_path + 't10k-labels-idx1-ubyte'
    ) 
    

    A_train = X_train[:n_train_samples].reshape((n_train_samples, 28, 28))
    A_test = X_test[:n_test_samples].reshape((n_test_samples, 28, 28))

    for i in range(n_train_samples):
        outpath = A_path_train + 'mnist_train_{}'.format(i) + '.png'
        imageio.imwrite(outpath, A_train[i])
    
    for i in range(n_test_samples):
        outpath = A_path_test + 'mnist_test_{}'.format(i) + '.png'
        imageio.imwrite(outpath, A_test[i])


    # mnist c
    path_train = 'data/mnistc/mnist_c/' + version + '/train_images.npy'
    path_test = 'data/mnistc/mnist_c/' + version + '/test_images.npy'

    train_data = np.load(path_train)
    test_data = np.load(path_test)

    train_data = train_data[:n_train_samples]
    test_data = test_data[:n_test_samples]

    for i in range(n_train_samples):
        path = B_path_train + version + '_train_{}'.format(i) + '.png'
        imageio.imwrite(path, train_data[i])

    for i in range(n_test_samples):
        path = B_path_test + version + '_test_{}'.format(i) + '.png'
        imageio.imwrite(path, test_data[i])


def get_data_h2z():
    # downloads horse2zebra images
    folder_name = "horse2zebra"
    path = "./data/"
    url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip"

    #display download progress 
    #https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html
    def reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                        (percent, progress_size / (1024 * 1024), speed, duration))
        sys.stdout.flush()

    def get_data(url,path, folder_name):
        new_path = path + folder_name

        if not os.path.exists(path):
            print("creating data folder")
            os.makedirs(path)
    
        if not os.path.exists(new_path):        
            print("creating zebra folder and downloading data")
        
            # download zipfile from url and store it in path
            urllib.request.urlretrieve(url, path+ "temp.zip",reporthook)
            print("unzipping files")
            # unzip file
            with zipfile.ZipFile(path +"temp.zip","r") as zip_ref:
                zip_ref.extractall(path)

            os.remove(path+"temp.zip")
            
            print("\nfinished download")
        else:
            print("data folder already there")


def create_batch(iterators, big_bs):
    images = []
    labels = []
    for i in range(big_bs):
        idx = np.random.randint(0,len(iterators))
        images += next(iterators[idx]).unsqueeze(dim =0)
        labels += [idx] 
    return torch.cat(images), torch.Tensor(np.array(labels))


def plot_predictions(original, generated, org_percentages, gen_percentages,o_attr_A = None, o_attr_B= None, g_attr_A= None, g_attr_B= None ):
    original  = original.detach()
    generated = generated.detach()

    if o_attr_A is None:
        plt.figure(figsize=(18,5))

        plt.subplot(1, 2, 1)
        plt.title("Original image \n Classified as {} %  horse, {} % zebra".format(org_percentages[0], org_percentages[1]))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(original.squeeze().permute(1,2,0).numpy())

        plt.subplot(1, 2, 2)
        plt.title("Generated image \n Classified as {} %  horse, {} % zebra".format(gen_percentages[0], gen_percentages[1]))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(generated.squeeze().permute(1,2,0).numpy())

    else:
        plt.figure(figsize=(18,10))  
        
        plt.subplot(2, 3, 1)
        plt.title("Original image \n Classified as {} %  horse, {} % zebra".format(org_percentages[0], org_percentages[1]))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(original.squeeze().permute(1,2,0).numpy())

        plt.subplot(2, 3, 2)
        plt.title("Relevance for Domain A")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(o_attr_A*250)

        plt.subplot(2, 3, 3)
        plt.title("Relevance for Domain B")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(o_attr_B*250)

        plt.subplot(2, 3, 4)
        plt.title("Generated image \n Classified as {} %  horse, {} % zebra".format(gen_percentages[0], gen_percentages[1]))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(generated.squeeze().permute(1,2,0).numpy())

        plt.subplot(2, 3, 5)
        plt.title("Relevance for Domain A")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(g_attr_A*250)

        plt.subplot(2, 3, 6)
        plt.title("Relevance for Domain B")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(g_attr_B*250)


def create_explanation(original, generated, model, algorithm= "integrated gradients"):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    original.requires_grad=True 
    original = original.to(device)

    generated.requires_grad=True 
    generated = generated.to(device)

    o_baseline = torch.zeros(original.shape).to(device)
    g_baseline = torch.zeros(generated.shape).to(device)

    if algorithm == "integrated gradients":
        ig  = IntegratedGradients(model)
 
        o_attributions_A, delta = ig.attribute(original, o_baseline, target=0, return_convergence_delta=True)
        o_attributions_B, delta = ig.attribute(original, o_baseline, target=1, return_convergence_delta=True)
        o_attributions_A = o_attributions_A.detach().squeeze().permute(1,2,0).cpu().numpy()
        o_attributions_B = o_attributions_B.detach().squeeze().permute(1,2,0).cpu().numpy()

        g_attributions_A, delta = ig.attribute(generated, g_baseline, target=0, return_convergence_delta=True)
        g_attributions_B, delta = ig.attribute(generated, g_baseline, target=1, return_convergence_delta=True)
        g_attributions_A = g_attributions_A.detach().squeeze().permute(1,2,0).cpu().numpy()
        g_attributions_B = g_attributions_B.detach().squeeze().permute(1,2,0).cpu().numpy()
    
    elif algorithm == "saliency":
        sal = Saliency(model)
        o_attributions_A = sal.attribute(original, target = 0, abs=False).detach().squeeze().permute(1,2,0).cpu().numpy()
        o_attributions_B = sal.attribute(original, target = 1, abs=False).detach().squeeze().permute(1,2,0).cpu().numpy()

        g_attributions_A = sal.attribute(generated, target = 0, abs=False).detach().squeeze().permute(1,2,0).cpu().numpy()
        g_attributions_B = sal.attribute(generated, target = 1, abs=False).detach().squeeze().permute(1,2,0).cpu().numpy()

    return o_attributions_A, o_attributions_B, g_attributions_A, g_attributions_B