#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#$ -l cuda=1 # remove this line when no GPU is needed!
#$ -q all.q # do not fill the qlogin queue
#$ -cwd # start processes in current working directory
#$ -V # provide environment variables to processes

#Cluster settings, 
import os

import importlib
#import models
#import functions 
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
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
databaseName = "maps"
root_path_data = "./data/"+databaseName
root_path_checkpoints = "./checkpoints/"+databaseName

cycle  = models.CycleGAN(genA2B, genB2A, discA, discB, classifier, device, root_path_data, root_path_checkpoints)

param_train_cycle  = models.Param(channels = 3, epochs = 2, size= 256,  name ="cycle_test")


targetEpoch = 20
saveAt = 1 #how often to store the model (stores if epoch % saveAt == 0 || epoch == n_epochs-1 )

param_train1  = models.Param(channels = 3, epochs = targetEpoch, saveEpoch = saveAt, size= 256,  name ="cycle_r9_advMSE_l10", resnet_blocks=9, loss_adv=torch.nn.MSELoss(), lambdas=(10,0.5)) # default
param_train2  = models.Param(channels = 3, epochs = targetEpoch, saveEpoch = saveAt, size= 256,  name ="cycle_r9_advMSE_l5" , resnet_blocks=9, loss_adv=torch.nn.MSELoss(), lambdas=(5,0.5))
param_train3  = models.Param(channels = 3, epochs = targetEpoch, saveEpoch = saveAt, size= 256,  name ="cycle_r9_advL1_l10", resnet_blocks=9, loss_adv=torch.nn.L1Loss(), lambdas=(10,0.5))
param_train4  = models.Param(channels = 3, epochs = targetEpoch, saveEpoch = saveAt, size= 256,  name ="cycle_r9_advL1_l5" , resnet_blocks=9, loss_adv=torch.nn.L1Loss(), lambdas=(5,0.5))
param_train5  = models.Param(channels = 3, epochs = targetEpoch, saveEpoch = saveAt, size= 256,  name ="cycle_r5_advMSE_l10", resnet_blocks=5, loss_adv=torch.nn.MSELoss(), lambdas=(10,0.5))
param_train6  = models.Param(channels = 3, epochs = targetEpoch, saveEpoch = saveAt, size= 256,  name ="cycle_r5_advMSE_l5" , resnet_blocks=5, loss_adv=torch.nn.MSELoss(), lambdas=(5,0.5))
param_train7  = models.Param(channels = 3, epochs = targetEpoch, saveEpoch = saveAt, size= 256,  name ="cycle_r5_advL1_l10", resnet_blocks=5, loss_adv=torch.nn.L1Loss(), lambdas=(10,0.5))
param_train8  = models.Param(channels = 3, epochs = targetEpoch, saveEpoch = saveAt, size= 256,  name ="cycle_r5_advL1_l5" , resnet_blocks=5, loss_adv=torch.nn.L1Loss(), lambdas=(5,0.5))
param_train_cycle_list = [param_train1, param_train2, param_train3, param_train4, param_train5, param_train6, param_train7, param_train8]

param_eval_testset = models.Param(channels = 3, size= 256) 
param_train_classifier = models.Param(channels = 3, epochs = 2, size = 256, name = "classifier_test")


print("generating eval sets")
for paramItem in param_train_cycle_list:
    #load model
    cycle.load_cycle_nets(epoch = targetEpoch, model_name = paramItem.name)
    print(device)
    #generate for every test image some output image
    cycle._create_evalset_paired(paramItem,subfolder=paramItem.name) 

#MSE? SSIM? FCN? 


print("calculating MSE errors")
losses= {}
for paramItem in param_train_cycle_list:
    source_domains = ["A", "B"]
    subfolder = paramItem.name
    losses[paramItem.name]={}
    for source in source_domains:  
        losses[paramItem.name][source]=[]
        folder=root_path_data + "/eval{}/{}/".format(source,subfolder)
        folderSize = int(len(os.listdir(folder)) / 3)
        for i in range(0,folderSize):
            gen="{}{}_generated.jpg".format(folder, i)
            exp="{}{}_expected.jpg".format(folder, i)
            gen=Image.open(gen)
            exp=Image.open(exp)
            loss=functions.calcMSE(gen,exp)
            losses[paramItem.name][source].append(loss)

for paramItem in param_train_cycle_list:
    source_domains = ["A", "B"]
    for source in source_domains:
        summ = np.array(losses[paramItem.name][source]).sum()
        std = np.std(losses[paramItem.name][source])
        overall = np.array(losses[paramItem.name]['A']).sum() + np.array(losses[paramItem.name]['B']).sum()
        print("param:{} \t source:{} \t std:{:.4f} \t sum:{:.2f} \t overall:{:.2f}".format(paramItem.name,source,std,summ,overall))
print("done.")
    