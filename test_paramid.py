#!/usr/bin/env python3
#-*- coding:utf-8 -*-
#$ -l cuda=1 # remove this line when no GPU is needed!
#$ -q all.q # do not fill the qlogin queue
#$ -cwd # start processes in current working directory
#$ -V # provide environment variables to processes
#$ -t 1-8 # start 8 instances: to train different models in parallel
#Cluster settings, 
try:
    model_param_id = int(os.environ['SGE_TASK_ID'])
    print("starting task with model_param_id: %s"%model_param_id)
except:
    print("no SGE_TASK_ID set, choosing default model parameters ")
    model_param_id = 0 #param_train_cycle_list[0] should be default model params
    
import os
import time
import torch

try:
	the_task = int(os.environ['SGE_TASK_ID'])
    print("starting task with the_task: %s"%the_task)
except:
	print("wird nicht parallel ausgefuehrt..!")
	the_task = 0

#test2.makesomefile(the_task)

#schreibe in eigene datei
file_blub = open('some%d.txt'%the_task,"w+")
file_blub.write("\ntorch version %s cuda: %s"%(torch.__version__,torch.cuda.is_available()))
file_blub.close()
print(torch.__version__)
print('I am a job task with ID %d.'%the_task)