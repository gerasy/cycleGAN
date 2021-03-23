import os
import urllib
import zipfile

from os import listdir
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import models as pytroch_models

import numpy as np

# import cv2
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

import models


##################### download data

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
            # print(download_path)
            # print(save_path)
            print('Downloading {}'.format(r))
            try:
                urlretrieve(download_path, save_path)
            except:
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
            with gzip.open(zipped_path, 'rb') as zipped_file:
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
            if f.startswith('mnist_c/' + version):
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
        images_path=mnist_path + 'train-images-idx3-ubyte',
        labels_path=mnist_path + 'train-labels-idx1-ubyte'
    )

    X_test, y_test = loadlocal_mnist(
        images_path=mnist_path + 't10k-images-idx3-ubyte',
        labels_path=mnist_path + 't10k-labels-idx1-ubyte'
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


def get_data_h2z(dataBaseName="horse2zebra"):  # donedbn
    # downloads horse2zebra images

    path = "./data/"
    url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/" + dataBaseName + ".zip"
    get_data(url, path, dataBaseName)
    # display download progress
    # https://blog.shichao.io/2012/10/04/progress_speed_indicator_for_urlretrieve_in_python.html


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


def get_data(url, path, folder_name):
    new_path = path + folder_name

    if not os.path.exists(path):
        print("creating data folder")
        os.makedirs(path)

    if not os.path.exists(new_path):
        print("creating zebra folder and downloading data")

        # download zipfile from url and store it in path
        urllib.request.urlretrieve(url, path + "temp.zip", reporthook)
        print("unzipping files")
        # unzip file
        with zipfile.ZipFile(path + "temp.zip", "r") as zip_ref:
            zip_ref.extractall(path)

        os.remove(path + "temp.zip")

        print("\nfinished download")
    else:
        print("data folder already there")


##################### helping functions for classifier training

def create_batch(iterators, big_bs):
    images = []
    labels = []
    for i in range(big_bs):
        idx = np.random.randint(0, len(iterators))
        images += next(iterators[idx]).unsqueeze(dim=0)
        labels += [idx]
    return torch.cat(images), torch.Tensor(np.array(labels))


##################### plotting/ calculating  - classifier related

def plot_predictions(original, generated, org_percentages, gen_percentages, o_attr_A=None, o_attr_B=None, g_attr_A=None,
                     g_attr_B=None):
    original = original.detach()
    generated = generated.detach()

    if o_attr_A is None:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.title(
            "Original image \n Classified as {} %  horse, {} % zebra".format(org_percentages[0], org_percentages[1]))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(original.squeeze().permute(1, 2, 0).numpy())

        plt.subplot(1, 2, 2)
        plt.title(
            "Generated image \n Classified as {} %  horse, {} % zebra".format(gen_percentages[0], gen_percentages[1]))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(generated.squeeze().permute(1, 2, 0).numpy())

        # plt.savefig("./image_output/paper_background_fail.jpg")

    else:
        plt.figure(figsize=(18, 10))

        plt.subplot(2, 3, 1)
        plt.title(
            "Original image \n Classified as {} %  horse, {} % zebra".format(org_percentages[0], org_percentages[1]))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(original.squeeze().permute(1, 2, 0).numpy())

        plt.subplot(2, 3, 2)
        plt.title("Relevance for Domain A")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(o_attr_A * 250)

        plt.subplot(2, 3, 3)
        plt.title("Relevance for Domain B")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(o_attr_B * 250)

        plt.subplot(2, 3, 4)
        plt.title(
            "Generated image \n Classified as {} %  horse, {} % zebra".format(gen_percentages[0], gen_percentages[1]))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(generated.squeeze().permute(1, 2, 0).numpy())

        plt.subplot(2, 3, 5)
        plt.title("Relevance for Domain A")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(g_attr_A * 250)

        plt.subplot(2, 3, 6)
        plt.title("Relevance for Domain B")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(g_attr_B * 250)


def create_explanation(original, generated, model, algorithm="integrated gradients"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original.requires_grad = True
    original = original.to(device)

    generated.requires_grad = True
    generated = generated.to(device)

    o_baseline = torch.zeros(original.shape).to(device)
    g_baseline = torch.zeros(generated.shape).to(device)

    if algorithm == "integrated gradients":
        ig = IntegratedGradients(model)

        o_attributions_A, delta = ig.attribute(original, o_baseline, target=0, return_convergence_delta=True)
        o_attributions_B, delta = ig.attribute(original, o_baseline, target=1, return_convergence_delta=True)
        o_attributions_A = o_attributions_A.detach().squeeze().permute(1, 2, 0).cpu().numpy()
        o_attributions_B = o_attributions_B.detach().squeeze().permute(1, 2, 0).cpu().numpy()

        g_attributions_A, delta = ig.attribute(generated, g_baseline, target=0, return_convergence_delta=True)
        g_attributions_B, delta = ig.attribute(generated, g_baseline, target=1, return_convergence_delta=True)
        g_attributions_A = g_attributions_A.detach().squeeze().permute(1, 2, 0).cpu().numpy()
        g_attributions_B = g_attributions_B.detach().squeeze().permute(1, 2, 0).cpu().numpy()

    elif algorithm == "saliency":
        sal = Saliency(model)
        o_attributions_A = sal.attribute(original, target=0, abs=False).detach().squeeze().permute(1, 2,
                                                                                                   0).cpu().numpy()
        o_attributions_B = sal.attribute(original, target=1, abs=False).detach().squeeze().permute(1, 2,
                                                                                                   0).cpu().numpy()

        g_attributions_A = sal.attribute(generated, target=0, abs=False).detach().squeeze().permute(1, 2,
                                                                                                    0).cpu().numpy()
        g_attributions_B = sal.attribute(generated, target=1, abs=False).detach().squeeze().permute(1, 2,
                                                                                                    0).cpu().numpy()

    return o_attributions_A, o_attributions_B, g_attributions_A, g_attributions_B


def get_prediction_matrix(model, param_list, pre_calculated=False, source_domain="A",
                          dataBaseName="horse2zebra"):  # donedbn
    if not pre_calculated:
        low_pic = 0
        high_pic = 119
        model.load_classifier()
        param_eval_testset = models.Param(channels=3, size=256)
        model.load_cycle_nets(epoch=100, model_name=param_list[0].name)
        org, gen, org_percentage, gen_percentage = model.eval_testset(param_eval_testset, source_domain=source_domain,
                                                                      pic_number_low=low_pic, pic_number_high=high_pic,
                                                                      pre_generated=False, plot=False, explain=False)

        idx, inv_idx = (0, 1) if source_domain == "A" else (1, 0)
        tup = (np.array(org_percentage)[:, idx][:, np.newaxis], np.array(gen_percentage)[:, inv_idx][:, np.newaxis])

        grid = np.concatenate(tup, axis=1)

        for i in range(1, len(param_list)):
            model.load_cycle_nets(epoch=100, model_name=param_list[i].name)
            org, gen, org_percentage, gen_percentage = model.eval_testset(param_eval_testset,
                                                                          source_domain=source_domain,
                                                                          pic_number_low=low_pic,
                                                                          pic_number_high=high_pic,
                                                                          pre_generated=False, plot=False,
                                                                          explain=False)

            grid = np.concatenate((grid, np.array(gen_percentage)[:, inv_idx][:, np.newaxis]), axis=1)

        np.save("./checkpoints/" + dataBaseName + "/prediction_matrix_{}".format(source_domain), grid)

    else:
        grid = np.load("./checkpoints/" + dataBaseName + "/prediction_matrix_{}.npy".format(source_domain),
                       allow_pickle=True)

    return grid


def plot_prediction_matrix(matrix, param_list, source_domain="A"):
    avg = avg = np.mean(matrix, axis=0)
    avg = avg[np.newaxis, :]

    wrong_count = (matrix < 50).sum(axis=0)
    wrong_count = wrong_count[np.newaxis, :]

    matrix = np.concatenate((matrix, avg), axis=0)
    matrix = np.concatenate((matrix, wrong_count), axis=0)

    plt.figure(figsize=(25, matrix.shape[0] // 2))
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False

    plt.imshow(matrix, interpolation='nearest', cmap="seismic_r", vmin=0, vmax=100)

    plt.title("prediction matrix for {}".format(source_domain))

    tick_marks_x = np.arange(len(param_list) + 1)
    tick_marks_y = np.arange(matrix.shape[0])
    names = ["original img"] + [param.name for param in param_list]
    img_idx = ["img_idx " + str(idx) for idx in np.arange(matrix.shape[0] - 2)] + ["average", "wrong count"]

    plt.xticks(tick_marks_x, names, rotation=45, ha="left")
    plt.yticks(tick_marks_y, img_idx)

    fmt = '.2f'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.xlabel("Predcition scores for classes. Left colum original image, other columns generated images")
    plt.tight_layout()
    plt.savefig("./image_output/prediction_matrix_{}.jpg".format(source_domain))


def plot_generated_images(model, img_idx, param_list, source_domain="A", dataBaseName="horse2zebra"):
    model.load_classifier()
    param_eval_testset = models.Param(channels=3, size=256)

    plt.figure(figsize=(27, 8))  # 5

    rows = 1
    cols = len(param_list) + 1

    model.load_cycle_nets(epoch=100, model_name=param_list[0].name)
    org, gen, org_percentage, gen_percentage = model.eval_testset(param_eval_testset, source_domain=source_domain,
                                                                  pic_number_low=img_idx, pic_number_high=img_idx,
                                                                  pre_generated=False, plot=False, explain=False)

    loss = calc_background_loss(model=model, source_domain=source_domain, img_idx=img_idx, masks_precalculated=True,
                                plot=False, dataBaseName=dataBaseName)
    plt.subplot(rows, cols, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title("original image \n pred. score: {}".format(org_percentage[0]))
    plt.imshow(org.squeeze().permute(1, 2, 0).numpy())

    plt.subplot(rows, cols, 2)
    plt.xticks([])
    plt.yticks([])
    plt.title(
        "generated image \n {} \n pred. score: {} \n background err: {}  ".format(param_list[0].name, gen_percentage[0],
                                                                                  loss))
    # plt.title("generated image \n background err.: {} \n {}".format(loss, "param. {}".format(1)))
    plt.imshow(gen.squeeze().permute(1, 2, 0).numpy())

    for i in range(1, len(param_list)):
        model.load_cycle_nets(epoch=100, model_name=param_list[i].name)
        org, gen, org_percentage, gen_percentage = model.eval_testset(param_eval_testset, source_domain=source_domain,
                                                                      pic_number_low=img_idx, pic_number_high=img_idx,
                                                                      pre_generated=False, plot=False, explain=False)
        loss = calc_background_loss(model=model, source_domain=source_domain, img_idx=img_idx, masks_precalculated=True,
                                    plot=False, dataBaseName=dataBaseName)
        plt.subplot(rows, cols, i + 2)
        plt.xticks([])
        plt.yticks([])
        plt.title("generated image \n {} \n pred. score: {} \n background err: {}  ".format(param_list[i].name,
                                                                                            gen_percentage[0], loss))
        # plt.title("generated image \n background err.: {} \n {}".format(loss, "param. {}".format(i+1))) #
        plt.imshow(gen.squeeze().permute(1, 2, 0).numpy())

    # plt.savefig("./image_output/paper_background_2.jpg")


##################### plotting/ calculating  - segmentation related

def get_background_loss_matrix(model, param_list, pre_calculated=False, source_domain="A", dataBaseName="horse2zebra",
                               epochToLoad=100):  # donedbn

    if not pre_calculated:

        # exclude samples from dataset where the segmentation algorithm fails to find correct segmentation. manually selected..
        if source_domain == "A":
            total_list = np.arange(0, 119)
            bad_list = [0, 3, 15, 24, 27, 30, 39, 42, 56, 74, 76, 79, 95, 97, 102, 109]

        elif source_domain == "B":
            total_list = np.arange(0, 140)
            bad_list = [23, 52, 118, 132, 134]

        good_list = [idx for idx in total_list if idx not in bad_list]

        COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # prepare segmentation model and generators
        model.segmentation.eval()
        model.load_cycle_nets(epoch=epochToLoad, model_name=param_list[0].name)

        # caclulate background losses for first parameter settings
        losses = []
        for idx in good_list:
            temp = calc_background_loss(model, source_domain=source_domain, img_idx=idx, masks_precalculated=True,
                                        plot=False, dataBaseName=dataBaseName)
            losses.append(temp)
        losses = np.array(losses)[:, np.newaxis]

        # calculate background losses for remaining paramter settings
        for i in range(1, len(param_list)):
            print("param number: ", i)
            model.load_cycle_nets(epoch=epochToLoad, model_name=param_list[i].name)

            temp_list = []
            for idx in good_list:
                temp = calc_background_loss(model, source_domain=source_domain, img_idx=idx, masks_precalculated=True,
                                            plot=False, dataBaseName=dataBaseName)
                temp_list.append(temp)

            losses = np.concatenate((losses, np.array(temp_list)[:, np.newaxis]), axis=1)

        np.save("./checkpoints/" + dataBaseName + "/background_loss_matrix_{}".format(source_domain), losses)

    else:
        # if background loss matrix is already calculated just load it
        losses = np.load("./checkpoints/" + dataBaseName + "/background_loss_matrix_{}.npy".format(source_domain),
                         allow_pickle=True)

    return losses


# calculates the background loss for a give image from a given domain
# background loss is the normalized difference of the non-target areas of the original and the generated image
def calc_background_loss(model, source_domain="A", img_idx=0, masks_precalculated=True, plot=True,
                         database_name="horse2zebra"):  # donedbn
    target_class = "horse" if source_domain == "A" else "zebra"
    net = model.genA2B if source_domain == "A" else model.genB2A

    path_org = "./data/" + database_name + "/eval{}/original_{}.jpg".format(source_domain, img_idx)

    # get mask
    if not masks_precalculated:
        mask = create_mask(model=model, source_domain=source_domain, img_idx=img_idx, plot=False, save=False)
    else:
        path_mask = "./data/" + database_name + "/eval{}/mask_{}.npy".format(source_domain, img_idx)
        mask = np.load(path_mask)

    # get original and generated image
    temp = tuple([0.5 for i in range(3)])  # param.channels

    little_t = t = [
        transforms.Resize(int(256 * 1.12), Image.BICUBIC),  # param.size
        transforms.ToTensor(),
        transforms.Normalize(temp, temp)]

    trans = transforms.Compose(little_t)

    unnorm = models.UnNormalize(mean=temp, std=temp)

    original = Image.open(path_org)
    original = trans(original).unsqueeze(dim=0)

    generated = net(original.clone().detach()).detach()

    original = unnorm(original)
    generated = unnorm(generated)

    # convert original and generated image to greyscale and apply the mask
    original = np.array(original.squeeze().permute(1, 2, 0))
    generated = np.array(generated.squeeze().permute(1, 2, 0))
    org = rgb2gray(original) * mask
    gen = rgb2gray(generated)[1:-1, 1:-1] * mask

    # clalculate normalized background loss
    l1 = torch.nn.L1Loss()
    normalzation_factor = np.sum(mask) / (mask.shape[0]) ** 2
    loss = l1(torch.tensor(org), torch.tensor(gen)).item() / normalzation_factor
    loss = np.round(loss, decimals=3)

    # plot original, generated and their masked versions
    if plot:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 4, 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(original)
        plt.title("background loss: {}".format(loss))
        plt.subplot(1, 4, 2)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(generated)
        plt.subplot(1, 4, 3)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(org, cmap="gray")
        plt.subplot(1, 4, 4)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(gen, cmap="gray")

    return loss


# takes the segments from get_prediction() and  joins every segment of the target class to one positive mask
# i.e. background all zeros, target all ones
def instance_segmentation(model, img_path, target, threshold=0.5):
    masks, pred_cls = get_prediction(model, img_path, threshold)
    mask_joint = torch.zeros((masks[0].shape))
    for i in range(len(masks)):
        if pred_cls[i] == target:
            mask_joint += masks[i]
    mask_joint[mask_joint > 1] = 1
    return mask_joint


# performs the actual segmentation, returns a list of found segments and their respective labels
def get_prediction(model, img_path, threshold):
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])  # kein normalize?
    img = transform(img)
    pred = model.segmentation([img])
    pred_score = list(pred[0]['scores'].detach().numpy())
    if np.max(pred_score) < 0.5:
        print("max pred_score {} is used, despide normal threshold is {}".format(np.max(pred_score), threshold))
        pred_t = [pred_score.index(np.max(pred_score))][-1]
    else:
        pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]

    if len(masks.shape) < 3:
        masks = masks[np.newaxis, :]
    masks = masks[:pred_t + 1]

    pred_class = pred_class[:pred_t + 1]
    return masks, pred_class


# function to convert rgb image to gray scale
def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


# function to calculate a negative mask for a given image from a given domain # i.e. background all ones, animal all zeros
def create_mask(model, source_domain="A", img_idx=0, plot=False, save=False, dataBaseName="horse2zebra",
                taget_class_A="horse", taget_class_B="zebra"):
    target_class = taget_class_A if source_domain == "A" else taget_class_B
    path_org = "./data/" + dataBaseName + "/eval{}/original_{}.jpg".format(source_domain, img_idx)
    mask = instance_segmentation(model, path_org, target=target_class)
    mask_np = mask.detach().numpy()
    mask = np.logical_xor(mask_np, np.ones(mask_np.shape))

    if save:
        np.save("./data/" + dataBaseName + "/eval{}/mask_{}".format(source_domain, img_idx), mask)
    if plot:
        plt.imshow(mask, cmap="gray")

    return mask


# function to plot the background loss matrix. similar to plot_prediction_matrix
def plot_backgorund_loss_matrix(matrix, param_list, source_domain="A"):
    if source_domain == "A":
        total_list = np.arange(0, 119)
        bad_list = [0, 3, 15, 24, 27, 30, 39, 42, 56, 74, 76, 79, 95, 97, 102, 109]

    elif source_domain == "B":
        total_list = np.arange(0, 140)
        bad_list = [23, 52, 118, 132, 134]

    good_list = [idx for idx in total_list if idx not in bad_list]

    avg = avg = np.mean(matrix, axis=0)
    avg = avg[np.newaxis, :]

    matrix = np.concatenate((matrix, avg), axis=0)

    plt.figure(figsize=(25, matrix.shape[0] // 2))

    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False

    plt.imshow(matrix, interpolation='nearest', cmap="seismic")  # , vmin= 0, vmax= 100

    plt.title("background loss matrix for source domain {}".format(source_domain))

    tick_marks_x = np.arange(len(param_list))
    tick_marks_y = np.arange(matrix.shape[0])
    names = [param.name for param in param_list]
    img_idx = ["img_idx " + str(idx) for idx in good_list] + ["average"]

    plt.xticks(tick_marks_x, names, rotation=45, ha="left")
    plt.yticks(tick_marks_y, img_idx)

    fmt = '.2f'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.xlabel("Background losses per image and parameter setting")
    plt.tight_layout()
    plt.savefig("./image_output/background_loss_matrix_{}.jpg".format(source_domain))


#### function for measuring locality: masked combined with  classifier + caputum

def estimate_relevance_location(model, source_domain="A", img_idx=0, masks_precalculated=True, plot=True,
                                dataBaseName="horse2zebra"):
    target_class = "horse" if source_domain == "A" else "zebra"
    net = model.genA2B if source_domain == "A" else model.genB2A

    path_org = "./data/" + dataBaseName + "/eval{}/original_{}.jpg".format(source_domain, img_idx)

    # get mask
    if not masks_precalculated:
        mask = create_mask(model=model, source_domain=source_domain, img_idx=img_idx, plot=False, save=False)
    else:
        path_mask = "./data/" + dataBaseName + "/eval{}/mask_{}.npy".format(source_domain, img_idx)
        mask = np.load(path_mask)

    # get original and generated image
    temp = tuple([0.5 for i in range(3)])  # param.channels

    little_t = t = [
        transforms.Resize(int(256 * 1.12), Image.BICUBIC),  # param.size
        transforms.ToTensor(),
        transforms.Normalize(temp, temp)]

    trans = transforms.Compose(little_t)

    unnorm = models.UnNormalize(mean=temp, std=temp)

    original = Image.open(path_org)
    original = trans(original).unsqueeze(dim=0)

    generated = net(original.clone().detach()).detach()

    # calculate attribution maps
    o_attributions_A, o_attributions_B, g_attributions_A, g_attributions_B = create_explanation(original, generated,
                                                                                                model.classifier,
                                                                                                algorithm="saliency")

    # convert attribution maps to gray scale (and crop them to correct shape)
    o_attributions_A = rgb2gray(o_attributions_A)
    o_attributions_B = rgb2gray(o_attributions_B)
    g_attributions_A = rgb2gray(g_attributions_A)[1:-1, 1:-1]
    g_attributions_B = rgb2gray(g_attributions_B)[1:-1, 1:-1]

    # claculate attribution ratio
    ratio_o_A = calc_attribution_ratio(o_attributions_A, mask)
    ratio_o_B = calc_attribution_ratio(o_attributions_B, mask)
    ratio_g_A = calc_attribution_ratio(g_attributions_A, mask)
    ratio_g_B = calc_attribution_ratio(g_attributions_B, mask)

    # unnormalize original an generated image, so they can be displayed
    original = unnorm(original.clone().detach()).squeeze().permute(1, 2, 0).numpy()
    generated = unnorm(generated.clone().detach()).squeeze().permute(1, 2, 0).numpy()

    if plot:
        # use this amplification_factor to scale atttributions, so they can be better plottet 
        amp = 250

        plt.figure(figsize=(18, 12))

        plt.subplot(3, 5, 1)
        plt.title("original image")
        plt.imshow(original)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 5, 2)
        plt.title("o_attributions_A")
        plt.imshow(o_attributions_A * amp, cmap="seismic", vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 5, 3)
        plt.title("o_attributions_B")
        plt.imshow(o_attributions_B * amp, cmap="seismic", vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 5, 4)
        plt.title("masekd o_attributions_A \n attribution ratio: {}".format(ratio_o_A))
        plt.imshow(o_attributions_A * mask * amp, cmap="seismic", vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 5, 5)
        plt.title("masked o_attributions_B \n attribution ratio: {}".format(ratio_o_B))
        plt.imshow(o_attributions_B * mask * amp, cmap="seismic", vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 5, 6)
        plt.title("generated")
        plt.imshow(generated)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 5, 7)
        plt.title("g_attributions_A")
        plt.imshow(g_attributions_A * amp, cmap="seismic", vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 5, 8)
        plt.title("g_attributions_B")
        plt.imshow(g_attributions_B * amp, cmap="seismic", vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 5, 9)
        plt.title("masked g_attributions_A \n attribution ratio: {}".format(ratio_g_A))
        plt.imshow(g_attributions_A * mask * amp, cmap="seismic", vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 5, 10)
        plt.title("masked g_attributions_B \n attribution ratio: {}".format(ratio_g_B))
        plt.imshow(g_attributions_B * mask * amp, cmap="seismic", vmin=-1, vmax=1)
        plt.xticks([])
        plt.yticks([])

        plt.subplot(3, 5, 11)
        plt.title("mask")
        plt.imshow(mask, cmap="gray")
        plt.xticks([])
        plt.yticks([])

        # plt.savefig("./image_output/paper_attr_low.jpg")
    return ratio_o_A, ratio_o_B, ratio_g_A, ratio_g_B


def calc_attribution_ratio(attribution_map, mask):
    inv_mask = np.logical_xor(mask, np.ones(mask.shape))
    temp_background = np.abs(attribution_map * mask)
    temp_target = np.abs(attribution_map * inv_mask)

    background_sum = np.sum(temp_background)
    target_sum = np.sum(temp_target)
    attribution_ratio = target_sum / background_sum

    # wtf funktioniert round erst, aber dann spaeter wird trotzdem der ungerundete wert geplottet? 
    # 
    test = np.copy(np.around(attribution_ratio, decimals=3))
    verzweiflung = int(test * 1000) / 1000
    return verzweiflung


def get_attribution_ratio_matrix(model, param_list, pre_calculated=False, source_domain="A", epochToLoad=100,
                                 dataBaseName="horse2zebra"):
    if not pre_calculated:

        # exclude samples from dataset where the segmentation algorithm fails to find correct segmentation. manually selected..
        if source_domain == "A":
            total_list = np.arange(0, 119)
            bad_list = [0, 3, 15, 24, 27, 30, 39, 42, 56, 74, 76, 79, 95, 97, 102, 109]

        elif source_domain == "B":
            total_list = np.arange(0, 140)
            bad_list = [23, 52, 118, 132, 134]

        good_list = [idx for idx in total_list if idx not in bad_list]

        model.load_cycle_nets(epoch=epochToLoad, model_name=param_list[0].name)

        # caclulate all attribution ratio for first parameter settings
        ratio_o_A = []
        ratio_o_B = []
        ratio_g_A = []
        ratio_g_B = []

        for idx in good_list:
            temp_tuple = estimate_relevance_location(model=model, source_domain=source_domain, img_idx=idx,
                                                     masks_precalculated=True, plot=False)
            ratio_o_A.append(temp_tuple[0])
            ratio_o_B.append(temp_tuple[1])
            ratio_g_A.append(temp_tuple[2])
            ratio_g_B.append(temp_tuple[3])

        ratio_o_A = np.array(ratio_o_A)[:, np.newaxis]
        ratio_o_B = np.array(ratio_o_B)[:, np.newaxis]
        ratio_g_A = np.array(ratio_g_A)[:, np.newaxis]
        ratio_g_B = np.array(ratio_g_B)[:, np.newaxis]

        # calculate attribution ratios for remaining paramter settings

        for i in range(1, len(param_list)):
            print("param number: ", i)
            model.load_cycle_nets(epoch=epochToLoad, model_name=param_list[i].name)

            temp_ratio_g_A = []
            temp_ratio_g_B = []

            for idx in good_list:
                temp_tuple = estimate_relevance_location(model=model, source_domain=source_domain, img_idx=idx,
                                                         masks_precalculated=True, plot=False)

                temp_ratio_g_A.append(temp_tuple[2])
                temp_ratio_g_B.append(temp_tuple[3])

            ratio_g_A = np.concatenate((ratio_g_A, np.array(temp_ratio_g_A)[:, np.newaxis]), axis=1)
            ratio_g_B = np.concatenate((ratio_g_B, np.array(temp_ratio_g_B)[:, np.newaxis]), axis=1)

        # concatenate all parts to get on big matrix
        temp_ratio_o = np.concatenate((ratio_o_A, ratio_o_B), axis=1)
        temp_ratio_g = np.concatenate((ratio_g_A, ratio_g_B), axis=1)
        attribution_ratio_matrix = np.concatenate((temp_ratio_o, temp_ratio_g), axis=1)

        np.save("./checkpoints/" + dataBaseName + "/attribution_ratio_matrix_{}".format(source_domain),
                attribution_ratio_matrix)

    else:  # if precalculated
        attribution_ratio_matrix = np.load(
            "./checkpoints/" + dataBaseName + "/attribution_ratio_matrix_{}.npy".format(source_domain),
            allow_pickle=True)

    return attribution_ratio_matrix


def plot_attribution_ratio_matrix(matrix, param_list, source_domain="A"):
    # exclude samples from dataset where the segmentation algorithm fails to find correct segmentation. manually selected..
    if source_domain == "A":
        total_list = np.arange(0, 119)
        bad_list = [0, 3, 15, 24, 27, 30, 39, 42, 56, 74, 76, 79, 95, 97, 102, 109]

    elif source_domain == "B":
        total_list = np.arange(0, 140)
        bad_list = [23, 52, 118, 132, 134]

    good_list = [idx for idx in total_list if idx not in bad_list]

    avg = np.mean(matrix, axis=0)
    avg = avg[np.newaxis, :]

    matrix = np.concatenate((matrix, avg), axis=0)

    plt.figure(figsize=(50, matrix.shape[0] // 2))  # (50,matrix.shape[0]//2))
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False

    plt.imshow(matrix, interpolation='nearest', cmap="seismic_r")  # , vmin= 0, vmax= 100

    plt.title("attribution ratio matrix for source domain {}".format(source_domain))

    tick_marks_x = np.arange(len(param_list) * 2 + 2)  # for every parameter there are 4 attribution scores
    tick_marks_y = np.arange(matrix.shape[0])

    temp = range(1, 9)
    t1 = ["o_A"]
    t2 = ["g_A_param{}".format(x) for x in temp]
    t3 = ["o_B"]
    t4 = ["g_B_param{}".format(x) for x in temp]

    names = t1 + t3 + t2 + t4
    img_idx = ["img_idx " + str(idx) for idx in good_list] + ["average"]

    plt.xticks(tick_marks_x, names, rotation=45, ha="left")
    plt.yticks(tick_marks_y, img_idx)

    fmt = '.2f'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.xlabel(
        "attribution scores for original and generated images w.r.t. to different classes. Source domain: {}".format(
            source_domain))
    plt.tight_layout()
    plt.savefig("./image_output/attribution_ratio_matrix_source_domain_{}.jpg".format(source_domain))


def calc_fid_scores(model, param_list):
    results = {}
    for param in param_list:
        model.load_cycle_nets(epoch=100, model_name=param.name)
        scoreA2B, scoreB2A = model.calc_fid(param)
        results[param.name] = (scoreA2B, scoreB2A)
    return results


def calcL1(img1, img2):
    temp = tuple([0.5 for i in range(3)])  # param.channels
    little_t = [
        transforms.Resize(int(256 * 1.12), Image.BICUBIC),  # param.size
        transforms.ToTensor(),
        transforms.Normalize(temp, temp)]

    trans = transforms.Compose(little_t)

    unnorm = models.UnNormalize(mean=temp, std=temp)

    img1 = trans(img1).unsqueeze(dim=0)
    img2 = trans(img2).unsqueeze(dim=0)

    img1 = unnorm(img1)
    img2 = unnorm(img2)

    # convert img1 and ịmg2 image to greyscale and apply the mask
    img1 = np.array(img1.squeeze().permute(1, 2, 0))
    img2 = np.array(img2.squeeze().permute(1, 2, 0))

    #print("img1.shape:{} img2.shape:{}".format(img1.shape, img2.shape))
    # clalculate normalized background loss
    l1 = torch.nn.L1Loss()
    loss = l1(torch.tensor(img1), torch.tensor(img2)).item()
    loss = np.round(loss, decimals=3)
    return loss

def calcMSE(img1, img2):
    temp = tuple([0.5 for i in range(3)])  # param.channels
    little_t = [
        transforms.Resize(int(256 * 1.12), Image.BICUBIC),  # param.size
        transforms.ToTensor(),
        transforms.Normalize(temp, temp)]

    trans = transforms.Compose(little_t)

    unnorm = models.UnNormalize(mean=temp, std=temp)

    img1 = trans(img1).unsqueeze(dim=0)
    img2 = trans(img2).unsqueeze(dim=0)

    img1 = unnorm(img1)
    img2 = unnorm(img2)

    # convert img1 and ịmg2 image to greyscale and apply the mask
    img1 = np.array(img1.squeeze().permute(1, 2, 0))
    img2 = np.array(img2.squeeze().permute(1, 2, 0))

    #print("img1.shape:{} img2.shape:{}".format(img1.shape, img2.shape))
    # clalculate normalized background loss
    mse = torch.nn.MSELoss()
    loss = mse(torch.tensor(img1), torch.tensor(img2)).item()
    loss = np.round(loss, decimals=3)
    return loss
