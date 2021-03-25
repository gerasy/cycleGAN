#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#$ -l cuda=1 # remove this line when no GPU is needed!
#$ -q all.q # do not fill the qlogin queue
#$ -cwd # start processes in current working directory
#$ -V # provide environment variables to processes



import importlib
import models
import functions 
import torch
import torch.nn as nn
importlib.reload(models)
importlib.reload(functions)
import numpy as np
#different import way for cluster
'''
import imp
models = imp.load_source('models', './models.py')
functions = imp.load_source('functions', './functions.py')
'''

# init CycleGAN
genA2B = models.Generator(input_nc=3, output_nc=3, n_residual_blocks=9)
genB2A = models.Generator(input_nc=3, output_nc=3, n_residual_blocks=9)
discA  = models.Discriminator(input_nc=3)
discB  = models.Discriminator(input_nc=3)
classifier = models.Classifier().net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path_data = "./data/horse2zebra"
root_path_checkpoints = "./checkpoints/horse2zebra"

targetEpoch = 100

cycle  = models.CycleGAN(genA2B, genB2A, discA, discB, classifier, device, root_path_data, root_path_checkpoints)


# parameter comibination tested for this milestone
param_train1  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r9_advMSE_l10", resnet_blocks=9, loss_adv=torch.nn.MSELoss(), lambdas=(10,0.5)) # default
param_train2  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r9_advMSE_l5" , resnet_blocks=9, loss_adv=torch.nn.MSELoss(), lambdas=(5,0.5))
param_train3  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r9_advL1_l10", resnet_blocks=9, loss_adv=torch.nn.L1Loss(), lambdas=(10,0.5))
param_train4  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r9_advL1_l5" , resnet_blocks=9, loss_adv=torch.nn.L1Loss(), lambdas=(5,0.5))
param_train5  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r5_advMSE_l10", resnet_blocks=5, loss_adv=torch.nn.MSELoss(), lambdas=(10,0.5))
param_train6  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r5_advMSE_l5" , resnet_blocks=5, loss_adv=torch.nn.MSELoss(), lambdas=(5,0.5))
param_train7  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r5_advL1_l10", resnet_blocks=5, loss_adv=torch.nn.L1Loss(), lambdas=(10,0.5))
param_train8  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r5_advL1_l5" , resnet_blocks=5, loss_adv=torch.nn.L1Loss(), lambdas=(5,0.5))
param_train_cycle_list = [param_train1, param_train2, param_train3, param_train4, param_train5, param_train6, param_train7, param_train8]

# paramter to load the background error verison. since this extension needed major changes in training/loaded training was done locally.
param_back_err = models.Param(name ="background_default", epochs = 80)

# paramters needed to call eval_testset() function
param_eval_testset = models.Param(channels = 3, size= 256) 

# parameters used for training the classifier
param_train_classifier = models.Param(channels = 3, epochs = targetEpoch, size = 256, name = "classifier_test")


res=functions.calc_fid_scores(model=cycle, param_list=param_train_cycle_list)

