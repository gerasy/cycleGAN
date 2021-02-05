#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#$ -l cuda=1 # remove this line when no GPU is needed!
#$ -q all.q # do not fill the qlogin queue
#$ -cwd # start processes in current working directory
#$ -V # provide environment variables to processes
#$ -t 1-8 # start 8 instances: to train different models in parallel
#Cluster settings, 
import os
try:
    model_param_id = int(os.environ['SGE_TASK_ID'])
    print("starting task with model_param_id: %s"%model_param_id)
except:
    print("no SGE_TASK_ID set, choosing default model parameters ")
    model_param_id = 0 #param_train_cycle_list[0] should be default model params

import importlib
#import models
#import functions 
import torch
import torch.nn as nn
#importlib.reload(models)
#importlib.reload(functions)

#different import way for cluster
import imp
models = imp.load_source('models', './models.py')
functions = imp.load_source('functions', './functions.py')

# init CycleGAN
genA2B = models.Generator(input_nc=3, output_nc=3, n_residual_blocks=9)
genB2A = models.Generator(input_nc=3, output_nc=3, n_residual_blocks=9)
discA  = models.Discriminator(input_nc=3)
discB  = models.Discriminator(input_nc=3)
classifier = models.Classifier().net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
root_path_data = "./data/horse2zebra"
root_path_checkpoints = "./checkpoints/horse2zebra"

cycle  = models.CycleGAN(genA2B, genB2A, discA, discB, classifier, device, root_path_data, root_path_checkpoints)

param_train_cycle  = models.Param(channels = 3, epochs = 2, size= 256,  name ="cycle_test")


targetEpoch = 100

param_train1  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r9_advMSE_l10", resnet_blocks=9, loss_adv=torch.nn.MSELoss(), lambdas=(10,0.5)) # default
param_train2  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r9_advMSE_l5" , resnet_blocks=9, loss_adv=torch.nn.MSELoss(), lambdas=(5,0.5))
param_train3  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r9_advL1_l10", resnet_blocks=9, loss_adv=torch.nn.L1Loss(), lambdas=(10,0.5))
param_train4  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r9_advL1_l5" , resnet_blocks=9, loss_adv=torch.nn.L1Loss(), lambdas=(5,0.5))
param_train5  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r5_advMSE_l10", resnet_blocks=5, loss_adv=torch.nn.MSELoss(), lambdas=(10,0.5))
param_train6  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r5_advMSE_l5" , resnet_blocks=5, loss_adv=torch.nn.MSELoss(), lambdas=(5,0.5))
param_train7  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r5_advL1_l10", resnet_blocks=5, loss_adv=torch.nn.L1Loss(), lambdas=(10,0.5))
param_train8  = models.Param(channels = 3, epochs = targetEpoch, size= 256,  name ="cycle_r5_advL1_l5" , resnet_blocks=5, loss_adv=torch.nn.L1Loss(), lambdas=(5,0.5))
param_train_cycle_list = [param_train1, param_train2, param_train3, param_train4, param_train5, param_train6, param_train7, param_train8]

param_eval_testset = models.Param(channels = 3, size= 256) 
param_train_classifier = models.Param(channels = 3, epochs = 2, size = 256, name = "classifier_test")

### test functions
''' scheint zu funktionieren, aber nicht sonderlich fuer deepnote geeigent'''
#load the model_param_id parameters (used for parallel training on cluster)
param_train_cycle = param_train_cycle_list[model_param_id-1]
cycle.train(param_train_cycle)   
sys.exit("Training done")     
