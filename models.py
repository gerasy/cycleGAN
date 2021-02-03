import os
import urllib
import zipfile

from os import listdir
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.models as offShelfModels
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

import functions


class CycleGAN(object):
    def __init__(self, genA2B, genB2A, discA, discB, classifier, device, root_path_data, root_path_checkpoints):
        self.genA2B = genA2B
        self.genB2A = genB2A
        self.discA = discA
        self.discB = discB
        self.classifier = classifier
        self.device = device
        self.cycle_trained = False
        self.classifier_trained = False
        self.root_path_data = root_path_data
        self.root_path_checkpoints = root_path_checkpoints
        

    def train(self,param):

        self.genA2B.apply(self._init_weights).to(self.device)
        self.genB2A.apply(self._init_weights).to(self.device)
        self.discA.apply(self._init_weights).to(self.device)
        self.discB.apply(self._init_weights).to(self.device)

        netG_A2B = self.genA2B
        netG_B2A = self.genB2A
        netD_A   = self.discA
        netD_B   = self.discB
        
        name = param.name
        input_nc = param.channels       # input channels
        output_nc = param.channels      # output channels
        epoch = 0                       # starting epoch
        n_epochs = param.epochs         # number of epochs of training
        decay_epoch = np.ceil(n_epochs / param.lr_sched) # epoch to start linearly decaying the learning rate 
        lr = param.lr                   # initial learning rate
        size = param.size               # image size (width or height), squared assumed
        batchSize = param.bs            # batchsize
        lambda_iden = param.lambdas[1]  # weighting factor for identity loss
        lambda_cyc  = param.lambdas[0]  # weighting factor for cycle loss
        size_replay_buffer = param.size_replay_buffer
        resnet_blocks = param.resnet_blocks
        loss_adv = param.loss_adv
        loss_cyc_ide = param.loss_cyc_ide
        down_upsampling_layers = param.down_upsampling_layers

        pathA_Train = self.root_path_data + "/trainA/" # param.pathA_Train
        pathB_Train = self.root_path_data + "/trainB/" #param.pathB_Train

        device = self.device

        # inputs and targets memory allocation
        Tensor = torch.Tensor
        input_A = Tensor(batchSize, input_nc, size, size)
        input_B = Tensor(batchSize, output_nc, size, size)
        target_real = torch.ones((1),requires_grad=False).to(device) 
        target_fake = torch.zeros((1),requires_grad=False).to(device)

        # init replayBuffer
        fake_A_buffer = ReplayBuffer(param.size_replay_buffer)
        fake_B_buffer = ReplayBuffer(param.size_replay_buffer)

        #init losses
        losses = {"epoch": [], "adv_G_A2B": [],"adv_G_B2A": [],"adv_D_A": [], "adv_D_B": [], "cycle_loss": [], "identity_loss": []}
        # define lossfunctions
        criterion_GAN = param.loss_adv
        criterion_cycle = param.loss_cyc_ide
        criterion_identity = param.loss_cyc_ide

        # define optimizers
        optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=lr, betas=(0.5, 0.999))
        optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=lr, betas=(0.5, 0.999))

        # define learning rate schedulers
        lr_sched_G   = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
        lr_sched_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
        lr_sched_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
        
        # define transformations for data
        temp = tuple([0.5 for i in range(param.channels)])

        trans = [
            transforms.Resize(int(size*1.12), Image.BICUBIC), 
            transforms.RandomCrop(size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(temp, temp)
        ]
        
        # create dataloader
        train_dataset = ImageDataset(pathA_Train, pathB_Train, transforms_ = trans,  unaligned=True, rgb = False)
        train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle = True) 

        # putting all nets to training mode
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()    

        for epoch in range(epoch,n_epochs):
            tic = time.time()
            for i, batch in enumerate(train_loader):
                inner_tic = time.time()
                # model input
                real_A = Variable(input_A.copy_(batch['A'])).to(device)
                real_B = Variable(input_B.copy_(batch['B'])).to(device)

                ###### Train Generators A2B and B2A #####
                optimizer_G.zero_grad()

                # GAN (adversial) loss
                # aka loss is small, if disc thiks generated sample looks real
                fake_B = netG_A2B(real_A)
                loss_GAN_A2B = criterion_GAN(netD_B(fake_B), target_real.expand_as(netD_B(fake_B)))
                fake_A = netG_B2A(real_B)
                loss_GAN_B2A = criterion_GAN(netD_A(fake_A), target_real.expand_as(netD_A(fake_A)))
                
                gan_loss = loss_GAN_A2B + loss_GAN_B2A

                # Cycle loss
                # aka loss is small if recovered image is similar to original 
                recovered_A = netG_B2A(fake_B)
                loss_cycle_ABA = criterion_cycle(recovered_A, real_A) 
                recovered_B = netG_A2B(fake_A)
                loss_cycle_BAB = criterion_cycle(recovered_B, real_B) 
                
                cycle_loss = loss_cycle_ABA + loss_cycle_BAB
                
                # Identity loss
                # G_A2B(B) should equal B if real B is fed
                same_B = netG_A2B(real_B)
                loss_identity_B = criterion_identity(same_B, real_B) 
                same_A = netG_B2A(real_A)
                loss_identity_A = criterion_identity(same_A, real_A) 
                
                identity_loss = loss_identity_A +loss_identity_B   
            
                # Total loss
                loss_G = gan_loss + (identity_loss * lambda_iden) + (cycle_loss * lambda_cyc)
                loss_G.backward()

                optimizer_G.step()

                ###### Train Discriminator A ######
                optimizer_D_A.zero_grad()

                # Real loss
                pred_real = netD_A(real_A)
                loss_D_real = criterion_GAN(pred_real, target_real.expand_as(pred_real))

                # Fake loss using a image buffer
                fake_A = fake_A_buffer.push_and_pop(fake_A)
                pred_fake = netD_A(fake_A.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake.expand_as(pred_fake))

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake) / 2
                loss_D_A.backward()

                optimizer_D_A.step()

                ###### Train Discriminator B #####
                optimizer_D_B.zero_grad()

                # Real loss
                pred_real = netD_B(real_B)
                loss_D_real = criterion_GAN(pred_real, target_real.expand_as(pred_real))
                
                # Fake loss
                fake_B = fake_B_buffer.push_and_pop(fake_B)
                pred_fake = netD_B(fake_B.detach())
                loss_D_fake = criterion_GAN(pred_fake, target_fake.expand_as(pred_fake))

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake) / 2
                loss_D_B.backward()

                optimizer_D_B.step()

                inner_tac = time.time()
                
                ''' wollen wir (a) nach jedem batch oder (b) allen z.B. 100 batches oder (c) nur nach jeder epoche die losses speichern?
                losses["epoch"].append(epoch)
                losses["adv_G_A2B"].append(loss_GAN_A2B.detach().cpu().numpy())
                losses["adv_G_B2A"].append(loss_GAN_B2A.detach().cpu().numpy())
                losses["adv_D_A"].append(loss_D_A.detach().cpu().numpy())
                losses["adv_D_B"].append(loss_D_B.detach().cpu().numpy())
                losses["cycle_loss"].append(cycle_loss.detach().cpu().numpy())
                losses["identity_loss"].append(identity_loss.detach().cpu().numpy())
                '''

                print("batch {} done in {} seconds, cycle_loss:{}".format(i+1,np.round(inner_tac-inner_tic, decimals = 4),cycle_loss))

            # save the last model
            if (epoch==n_epochs-1) :
                self._save_model(model = netG_A2B, path = self.root_path_checkpoints, name = "netG_A2B" + param.name, epoch = param.epochs)
                self._save_model(model = netG_B2A, path = self.root_path_checkpoints, name = "netG_B2A" + param.name, epoch = param.epochs)
                self._save_model(model = netD_A  , path = self.root_path_checkpoints, name = "netD_A"   + param.name, epoch = param.epochs)
                self._save_model(model = netD_B  , path = self.root_path_checkpoints, name = "netD_B"   + param.name, epoch = param.epochs)


            # save losses per epoch
            losses["epoch"].append(epoch)
            losses["adv_G_A2B"].append(loss_GAN_A2B.detach().cpu().numpy())
            losses["adv_G_B2A"].append(loss_GAN_B2A.detach().cpu().numpy())
            losses["adv_D_A"].append(loss_D_A.detach().cpu().numpy())
            losses["adv_D_B"].append(loss_D_B.detach().cpu().numpy())
            losses["cycle_loss"].append(cycle_loss.detach().cpu().numpy())
            losses["identity_loss"].append(identity_loss.detach().cpu().numpy())
            # backup losses
            np.save(self.root_path_checkpoints + "/losses_{}.npy".format(param.name), losses)
                 
            tac = time.time()
            print("epoch {} of {} finished in {} seconds, cycle_loss: {}".format(epoch+1,n_epochs_T, 
                                                                                np.round(tac-tic, decimals = 3), (cycle_loss))) 
        
            # update learning rates
            lr_sched_G.step()
            lr_sched_D_A.step()
            lr_sched_D_B.step()


    def train_classifier(self, param):
        if self.classifier_trained == True:
            print("classifier already trained")
        else:
            input_nc = param.channels               # input channels
            epoch = 0                               # starting epoch
            n_epochs = param.epochs                  # number of epochs of training
            decay_epoch = np.ceil(n_epochs / 2.0)   # epoch to start linearly decaying the learning rate 
            lr = param.lr                           # initial learning rate
            img_size = param.size                   # image size (width / height), squared assumed
            bs = 1                                  # die bs am besten so lassen oder sonst vorher gruendlich drueber nachdenken ob es anders wirklich geht
            big_bs = 8                              # die hier kann man hoch setzen, wenn die rechenleistung es her gibt
            device = self.device

            # init network
            c = self.classifier

            # init weights and putting them to device
            c.apply(self._init_weights).to(device)

            # define lossfunction
            criterion =  nn.CrossEntropyLoss()

            # define optimizers
            optimizer_c = torch.optim.Adam(c.parameters(), lr=lr, betas=(0.5, 0.999))

            # define learning rate schedulers
            lr_sched_G = torch.optim.lr_scheduler.LambdaLR(optimizer_c, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step) 

            # define transformations for data # hier vllt auch ueber param?
            t = [
                transforms.Resize(int(img_size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(img_size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(tuple([0.5 for i in range(param.channels)]), tuple([0.5 for i in range(param.channels)]))
            ]

            loader_0 = DataLoader(SingleDomainImages(self.root_path_data + "/trainA/", transforms_ = t), batch_size=bs) 
            loader_1 = DataLoader(SingleDomainImages(self.root_path_data + "/trainB/", transforms_ = t), batch_size=bs) 

            for epoch in range(n_epochs):  
                tic = time.time()
                it_0 = iter(loader_0)
                it_1 = iter(loader_1)
                its = [it_0, it_1]
                for idx in range(np.maximum(len(it_0),len(it_1))//big_bs): 
                    
                    img, lab = functions.create_batch(its, big_bs)
                    images = img.to(device)
                    labels = lab.to(device)

                    optimizer_c.zero_grad()
                    outputs = c(images)
                    loss = criterion(outputs,labels.long())
                    loss.backward()
                    optimizer_c.step()
                lr_sched_G.step()
                print("epoch {} finished in {} seconds with a loss of: {}".format(epoch, time.time()-tic, loss))
            self._save_model(model = c, path = self.root_path_checkpoints, name = param.name, epoch = param.epochs)
        self.classifier_trained = True
        print('Finished Training')
        
    def load_classifier(self):
        # loads the pretrained classifier. 
        # keine parameter uebergeben, da nur dieses zur auswahl steht
        name = "classifier_h2z_resnet18_v2"
        self._load_model(model = self.classifier, path = self.root_path_checkpoints, name = name , epoch = 100)
        self.classifier_trained = True


    def load_cycle_nets(self,epoch):
        # loads the pretrained cycle_nets, i.e. the two generators and two discriminators. 
        
        self._load_model(model = self.genA2B, path = self.root_path_checkpoints, name = "netG_A2B" , epoch = 100)
        self._load_model(model = self.genB2A, path = self.root_path_checkpoints, name = "netG_B2A" , epoch = 100)
        self._load_model(model = self.discA , path = self.root_path_checkpoints, name = "netD_A"   , epoch = 100)
        self._load_model(model = self.discB , path = self.root_path_checkpoints, name = "netD_B"   , epoch = 100)

        self.cycle_trained = True 

    def _create_evalset(self, param, name):
    ### creates eval dataset from test data
        temp = tuple([0.5 for i in range(param.channels)]) 

        little_t = t = [
            transforms.Resize(int(param.size*1.12), Image.BICUBIC), 
            transforms.ToTensor(),
            transforms.Normalize(temp, temp)]

        test_loader_A = DataLoader(SingleDomainImages(self.root_path_data +"/testA/", transforms_ = little_t), batch_size=1)
        test_loader_B = DataLoader(SingleDomainImages(self.root_path_data +"/testB/", transforms_ = little_t), batch_size=1)
        
        device = self.device
        source_domains = ["A", "B"]
        for source in source_domains:    
            loader = test_loader_A if source == "A" else test_loader_B

            it = iter(loader)
            gen = self.genA2B if source == "A" else self.genB2A
            unorm = UnNormalize(mean=temp, std=temp)
            for i in range(len(loader)):
                img = next(it).to(device)

                img_copy = unorm(img.clone())
                img_copy = img_copy.squeeze().detach().cpu().permute(1,2,0).numpy()

                original = Image.fromarray((img_copy*255).astype(np.uint8))
                original.save(self.root_path_data + "/{}{}/original_{}.jpg".format(name,source, i)) 

                generated = gen(img).squeeze().detach()
                generated = unorm(generated)
                generated = generated.squeeze().detach().cpu().permute(1,2,0).numpy()
                generated =  Image.fromarray((generated*255).astype(np.uint8))
                generated.save(self.root_path_data + "/{}{}/generated_{}.jpg".format(name, source, i)) 
            print("finished domain {}".format(source))

    def eval_testset(self, param, source_domain = "A",pic_number_low = 0, pic_number_high = 5, plot = True, explain = False, algorithm = "integrated gradients" ): #bs = 1
        
        temp = tuple([0.5 for i in range(param.channels)]) 

        little_t = t = [
            transforms.Resize(int(param.size*1.12), Image.BICUBIC), 
            transforms.ToTensor(),
            transforms.Normalize(temp, temp)] 
        
        ### check if an eval dataset exists, if no -> create
        pathA = self.root_path_data + "/evalA"
        pathB = self.root_path_data + "/evalB"
            
        if not os.path.exists(pathA):
            print("creating save folder:", pathA, " and ", pathB)
            os.makedirs(pathA)
            os.makedirs(pathB)
            self._create_evalset(param, name = "eval")

        softy = nn.Softmax(dim = 1)    

        trans = transforms.Compose(little_t)

        unorm = UnNormalize(mean=temp, std=temp)

        if not self.classifier_trained:
            self.load_classifier()

        c = self.classifier

        file_number = len(os.listdir(self.root_path_data + "/eval{}".format(source_domain))) //2 -1
        if pic_number_low < 0 or pic_number_low >= pic_number_high or pic_number_high > file_number:
            print("please choose pic_numbers in range [0,{}]".format(file_number))
            return [],[],[],[]

        list_original  = []
        list_generated = []
        list_org_percentages = []
        list_gen_percentages = []

        for i in range(pic_number_low, pic_number_high+1):  # hier noch zu hohe werte abfangen
            original  = Image.open(self.root_path_data + "/eval{}/original_{}.jpg".format(source_domain, i))
            original  = trans(original).unsqueeze(dim=0)

            generated = Image.open(self.root_path_data + "/eval{}/generated_{}.jpg".format(source_domain, i))
            generated = trans(generated).unsqueeze(dim=0)

            pred_org = np.round(softy(c(original)).detach().numpy()[:,:2], decimals = 3)
            pred_gen = np.round(softy(c(generated)).detach().numpy()[:,:2], decimals = 3)

            original = unorm(original)
            generated = unorm(generated)
            
            org_percentage = [np.round(pred_org[0][0]*100, decimals = 2) ,np.round(pred_org[0][1]*100, decimals =2) ]
            gen_percentage = [np.round(pred_gen[0][0]*100, decimals = 2) ,np.round(pred_gen[0][1]*100, decimals =2) ]
            
            # storing results as list
            list_original.append(original)
            list_generated.append(generated)
            list_org_percentages.append(org_percentage)
            list_gen_percentages.append(gen_percentage)

            if plot:
                if explain:
                    o_attr_A, o_attr_B, g_attr_A, g_attr_B = functions.create_explanation(original, generated, model = c, algorithm = algorithm)
                    functions.plot_predictions(original, generated, org_percentage, gen_percentage, o_attr_A, o_attr_B, g_attr_A, g_attr_B)
                else:
                    functions.plot_predictions(original, generated, org_percentage, gen_percentage)
            
        return torch.cat(list_original), torch.cat(list_generated), list_org_percentages, list_gen_percentages

    #def continue_train():
        # hier richtige parameter wie aktuelle lr und so beachten
        # vllt doch nur eine train funktion fuer für das cycleGan die etwas flexibler ist, statt train und countiue train?
    
   
    def _init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    def _save_model(self, model, path, name, epoch):
        if not os.path.exists(path):
            print("creating save folder:", path)
            os.makedirs(path)
        torch.save(model.state_dict(), "{}/{} epoch{}".format(path, name, epoch))

    def _load_model(self, model, path, name, epoch):     
        try:
            if self.device == "cuda":
                model.load_state_dict(torch.load("{}/{} epoch{}".format(path, name, epoch)))
            else:
                model.load_state_dict(torch.load("{}/{} epoch{}".format(path, name, epoch),map_location=torch.device('cpu')))
            model.eval()
            print("model loaded:"+"{}/{} epoch{}".format(path, name, epoch))
        except Exception as e:
            if not os.path.exists(path):
                print("path doesnt exist!", path)
            print("An error occured while loading the model:\n"+"{}/{} epoch{}".format(path, name, epoch))
            print(e)


class Param():
    def __init__(self, channels = 3, epochs = 100, size = 256, name="default", down_upsampling_layers=2, lr = 0.0002, lr_sched = 2.0, size_replay_buffer = 50, resnet_blocks =9, loss_adv = torch.nn.MSELoss(), loss_cyc_ide = torch.nn.L1Loss() , lambdas=(10,0.5), bs=1):
        self.channels = channels 
        self.epochs = epochs     
        self.size = size         
        self.name = name         #which param and which value
        self.lr = lr             
        self.lr_sched = lr_sched
        self.size_replay_buffer = size_replay_buffer
        self.resnet_blocks = resnet_blocks   # (1)
        self.loss_adv = loss_adv             # (2)
        self.loss_cyc_ide = loss_cyc_ide
        self.lambdas = lambdas               # (3)
        self.bs = bs
        self.down_upsampling_layers = down_upsampling_layers  #


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Small_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=3, down_upsampling_layers = 2):
        super(Small_Generator, self).__init__()

        # Initial convolution block
        n = 32
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, n, 7),
                    nn.InstanceNorm2d(n),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        depth = down_upsampling_layers #not bigger than 4!
        in_features = n
        out_features = in_features*2
        for _ in range(depth):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(depth):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(n, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forwassrd(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        return  x 

class Small_Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Small_Discriminator, self).__init__()

        # A bunch of convolutions one after another
        n = 32
        model = [   nn.Conv2d(input_nc, n, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(n, 2*n, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(2*n), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [nn.Conv2d(2*n, 1, 2, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
       
        return  x 

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class ImageDataset(Dataset):
    def __init__(self, pathA, pathB,transforms_ = None, unaligned=False,rgb = True ):
        self.pathA = pathA
        self.pathB = pathB
        self.unaligned = unaligned
        self.rgb = rgb
        #dont do transformation if there are no transforms..
        if(transforms_==None):
            self.dontTransform = True
        else:
            self.transform = transforms.Compose(transforms_)
            self.dontTransform = False

    def __len__(self):
        return max(len(listdir(self.pathA)), len(listdir(self.pathB)))
    
        
    def __getitem__(self, index):
        sampleA = Image.open(self.pathA + listdir(self.pathA)[index % len(listdir(self.pathA))])
        if self.unaligned:
            sampleB = Image.open(self.pathB + listdir(self.pathB)[random.randint(0, len(listdir(self.pathB))-1)])
        else:
            sampleB = Image.open(self.pathB + listdir(self.pathB)[index % len(listdir(self.pathB))])
        
        #transform image AND convert to RGB to fix grayscale image dimension problem
        if self.rgb:
            sampleA = sampleA.convert('RGB')
            sampleB = sampleB.convert('RGB')
        #dont do transformation if there are no transforms..
        if not self.dontTransform:
            sampleA = self.transform(sampleA)
            sampleB = self.transform(sampleB)     
     
        return {'A': sampleA, 'B': sampleB}

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class Classifier():
    def __init__(self):
        c = offShelfModels.resnet18()
        c._modules["fc"] = nn.Linear(in_features = 512, out_features = 2, bias = True)
        # aplly weights necessary?
        self.net = c


class SingleDomainImages(Dataset):
    def __init__(self, root, transforms_=None, rgb = True):
        self.root = root
        self.files = os.listdir(self.root)            
        self.trans = transforms.Compose(transforms_)
        self.rgb = rgb
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        sample = Image.open(self.root + listdir(self.root)[index % len(listdir(self.root))])
        if self.rgb:
            sample = sample.convert('RGB')
        return self.trans(sample)


