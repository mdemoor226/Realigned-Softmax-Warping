import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from Model import WarpNet, WarpLoss
from Dataset import Data
from Utils import CalculateRecall, BalancedBatchSampler, CalculateNMI
from Utils import load_config, prepare_cub, prepare_cars, prepare_sop
from Logger import SetupLogger
import time
import datetime
import random
import json
import argparse

parser = argparse.ArgumentParser(description='Realigned Softmax Warping')
parser.add_argument('--dataset', default='cub')
parser.add_argument('--config', default='config.json')
parser.add_argument('--warp', default='True')
parser.add_argument('--trainval', default='True')
parser.add_argument('--noval', default='False')
parser.add_argument('--resume', default=None)

args = parser.parse_args()

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.enabled = False
#torch.backends.cudnn.deterministic = True

datasets = {
        'cub': prepare_cub,
        'cars': prepare_cars,
        'sop': prepare_sop,
        }

class Trainer(object):
    def __init__(self, cfg, dataset='cub', warp=True, trainval=True, noval=False, ckpt_path=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.epochs = cfg[dataset]['epochs']
        self.batch_size = cfg[dataset]['batch_size']
        self.n_classes = cfg[dataset]['classes_per_batch']
        self.n_samples = self.batch_size // self.n_classes

        #Prepare Dataset
        TrDataset, TsDataset = datasets[dataset](TrainVal=trainval)
        print("Using {} training samples.".format(len(TrDataset)))
        print("Using {} validation samples.".format(len(TsDataset)))
        
        #Train DataLoader
        Train_Data =  Data(TrDataset, cfg)
        TrSampler = BalancedBatchSampler(Train_Data, self.n_classes, self.n_samples, self.batch_size)
        self.TrLoader = data.DataLoader(Train_Data, batch_sampler=TrSampler, num_workers=2, pin_memory=True) 

        #Eval DataLoader
        Val_Data = Data(TsDataset, cfg, Train=False)
        self.ValLoader = data.DataLoader(Val_Data, batch_size=self.batch_size, num_workers=2, pin_memory=True, drop_last=False)
        
        Margin = cfg[dataset]['margin']
        self.model = WarpNet(dataset, pre_trained=cfg['pretrained']).to(self.device)
        self.Loss = WarpLoss(cfg, dataset, num_classes=len(np.unique(Train_Data.get_labels())), margin=Margin, warp=warp).to(self.device)
        if ckpt_path is not None:
            #Load Checkpoint
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            ProxyCheckpoint = "./Proxies/Proxy_" + ckpt_path[-20:]
            proxy_checkpoint = torch.load(ProxyCheckpoint, map_location=self.device)
            self.Loss.load_state_dict(proxy_checkpoint)
	
        Parameters = [{'params': [Param for Param in self.model.parameters()], 'weight_decay': 0.0000001, 'lr': cfg[dataset]['learning_rate']},
                         {'params': [Param for Param in self.Loss.parameters()], 'weight_decay': 0.0000001, 'lr': cfg[dataset]['proxy_learning_rate']}]
        self.optimizer = torch.optim.Adam(Parameters, eps=1.0)
        #self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        #Warm-Up Parameters
        if cfg['warmup']:
            Params = [param for param in self.model.parameters()]
            Warmup_Params = [{'params': Params[:-2], 'weight_decay': 0.0, 'lr': 0.0},
                             {'params': Params[-2:], 'weight_decay': 0.0, 'lr': 0.001},
                             {'params': [Param for Param in self.Loss.parameters()], 'weight_decay': 0.0, 'lr': 0.001}]
            self.warmup_opt = torch.optim.Adam(Warmup_Params, eps=1.0)
        
        #Some job tracking variables
        self.epoch = 0
        self.iteration = 0
        self.kVals = cfg[dataset]['recalls']
        self.eval_NMI = cfg[dataset]['eval_nmi']
        self.follow_cfg = cfg[dataset]['followup']
        self.noval = noval or not trainval
        self.epochs_per_eval = 10 if self.noval else self.epochs
        self.iters_per_log = cfg['iterations_per_log']
        self.epochs_per_ckpt = cfg['epochs_per_ckpt']
        self.init_eval = cfg['init_eval']
        self.warmup = cfg['warmup']
        self.warmup_epochs = 1 #Add to config later
        
        #Set the Logger so we know what's going on during the training process
        self.logger = SetupLogger(name="WarpNet", save_dir=".", distributed_rank=0, mode="a+")
        self.logger.info(
                "Hyperparameters:\nalpha: {:f} || Temp: {:.4f} || k1: {:.4f} || k2: {:.4f} || Margin: {:.4f} || lr: {} || plr: {} || f_alpha: {} || f_k1: {:.4f} || f_2: {:.4f} || f_temp: {} || flr: {} || fplr: {}".format(
        cfg[dataset]['alpha'], cfg[dataset]['temp'], cfg[dataset]['k1'], cfg[dataset]['k2'], Margin, cfg[dataset]['learning_rate'], cfg[dataset]['proxy_learning_rate'], self.follow_cfg['alpha'],
        self.follow_cfg['k1'], self.follow_cfg['k2'], self.follow_cfg['temp'], self.follow_cfg['learning_rate'], self.follow_cfg['proxy_learning_rate'])
        )
                    
    def _checkpoint(self, Recall):
        #Save Checkpoint
        Model = self.model
        Dir = os.path.expanduser("./Checkpoints/")
        if not os.path.exists(Dir):
            os.makedirs(Dir)
        
        file = "WarpNet_{}_{:.3f}Recall@1.pth".format(self.epoch,Recall)
        CHECKPOINT_PATH = os.path.join(Dir,file)
        torch.save(Model.state_dict(), CHECKPOINT_PATH)
    
        Loss = self.Loss
        LDir = os.path.expanduser("./Proxies/")
        if not os.path.exists(LDir):
            os.makedirs(LDir)
        
        file = "Proxy_{}_{:.3f}Recall@1.pth".format(self.epoch,Recall)
        ProxyCHECKPOINT_PATH = os.path.join(LDir,file)
        torch.save(Loss.state_dict(), ProxyCHECKPOINT_PATH)

    def _eval(self):
        self.model.eval()
        print("Evaluating Model...")
        
        #Begin Evaluation
        Eval = defaultdict(list)
        Model = self.model
        with torch.no_grad():
            print("Obtaining Validation Data Predictions...")
            for i,Sample in enumerate(self.ValLoader):
                #Send Data to GPU
                Batch = Sample['Data'].to(self.device)
                Labels = Sample['Labels'].to(self.device)
                
                #Forward through model
                Output = Model(Batch)
                
                #Gonvert Annotations to numpy arrays for post-processing
                IDs = Labels.cpu().numpy()
                Eval['Labels'].append(IDs[:,0])
                Eval['Preds'].append(Output.cpu().numpy())
            
            #Post-Process
            Preds = np.concatenate(Eval['Preds'], axis=0)
            TargetIDs = np.concatenate(Eval['Labels'], axis=0)
        
        #Collect Test Stats
        NMI = CalculateNMI(Preds, TargetIDs) if self.eval_NMI else 0.0
        print("NMI:",NMI)
        Recalls = CalculateRecall(Preds, TargetIDs, self.kVals)
        for k,R in zip(self.kVals,Recalls):
            print("Recall@{} for Validation data after epoch {}: {}".format(k, self.epoch, R))
        
        self.model.train()		
        return Recalls, NMI

    def warm_up(self):
        print("Warming up for {} epochs...".format(self.warmup_epochs))
        for epoch in range(self.warmup_epochs):
            for Sample in self.TrLoader:
                Batch = Sample['Data'].to(self.device)
                Labels = Sample['Labels'][:,0].to(self.device)
                Output = self.model(Batch)
                Loss = self.Loss(Output, Labels)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 10)
                #torch.nn.utils.clip_grad_value_(self.Loss.parameters(), 10)
                self.warmup_opt.zero_grad()
                Loss.backward()
                self.warmup_opt.step()
            
            print("{} epochs...".format(self.warmup_epochs - epoch - 1))
        self.model.train()
        print("Done.")
    
    def train(self):
        #Log the beginning of training
        self.logger.info("Training started for {:d} epochs.".format(self.epochs))
        self.model.train()
        
        #Warmup
        if self.warmup:
            self.warm_up()
        
        #Pre-Evaluation
        Recalls, NMI = [0,0,0,0], 0
        if self.init_eval:
            Recalls, NMI = self._eval()
        
        Max_Recall = Recalls[0]
        Best_Epoch = 1
        LossQueue = 100*[0.0]
        iters_per_epoch = int((self.batch_size/(self.n_classes*self.n_samples))*len(self.TrLoader))
        synchronize = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
        for epoch in range(1,self.epochs+1):
            self.epoch = epoch
            running_avg = 0
            
            synchronize()
            timecheck1 = time.time()
            for i,Sample in enumerate(self.TrLoader):
                self.iteration += 1
                #if self.iteration == 11:
                #    raise

                #Send Sample to GPU
                Batch = Sample['Data'].to(self.device)
                Labels = Sample['Labels'][:,0].to(self.device)
                
                #Forward
                Output = self.model(Batch)
                assert (Output == Output).all(), "Nans!"
                Loss = self.Loss(Output, Labels)
                
                #Running Average of Loss
                LossQueue[-1] = Loss.item()
                LossQueue = LossQueue[-1:] + LossQueue[:-1]
                running_avg = sum(LossQueue) / min(100,self.iteration)
                
                #Backward
                self.optimizer.zero_grad()
                Loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 10)
                self.optimizer.step()

                #Log some stuff
                if self.iteration % self.iters_per_log == 0:
                    synchronize()
                    timecheck2 = time.time()
                    timeleft = ((timecheck2 - timecheck1) / (i + 1))*(iters_per_epoch - i - 1)
                    eta = str(datetime.timedelta(seconds=int(timeleft)))
                    self.logger.info("Epoch: {:d}/{:d} eta: {} || Iteration: {:d} Lr: {:.6f} || ProxyLr: {:.4f} || Current Loss: {:.4f} || Most Recent Recall@1: {:.4f}".format(
                            epoch, self.epochs, eta,
                            self.iteration,
                            self.optimizer.param_groups[0]['lr'],
                            self.optimizer.param_groups[1]['lr'],
                            running_avg,
                            Recalls[0]))
            
            #self.lr_scheduler.step()
            
            #Run evalution if enough epochs have passed
            if epoch % self.epochs_per_eval == 0:
                Recalls, NMI = self._eval()
                
                #Log Result
                for k,R in zip(self.kVals,Recalls):
                    self.logger.info("Recall@{}: {:3f}, NMI: {:f}".format(k, R, NMI))
                
                if Recalls[0] > Max_Recall:
                    Max_Recall = Recalls[0]
                    Best_Epoch = epoch

            if epoch % self.epochs_per_ckpt == 0:
                self._checkpoint(Recalls[0])
            
            #TODO: Place this at top of loop?
            if epoch == self.epochs - self.follow_cfg['epochs']:
                self.Loss._UpdateParameters(self.follow_cfg)
                self.optimizer.param_groups[0]['lr'] = self.follow_cfg['learning_rate']
                self.optimizer.param_groups[1]['lr'] = self.follow_cfg['proxy_learning_rate']
                if self.noval:
                    self.epochs_per_eval = 5

        #Save the Checkpoint
        self._checkpoint(Recalls[0])
        
        #Training is finished
        self.logger.info("Best Epoch: {}".format(Best_Epoch))
        self.logger.info("Max Recall@1: {}".format(Max_Recall))
        self.logger.info("Trained for {:d} epochs. Goodbye.".format(self.epochs))
        #input("Press ENTER to continue...")

def main(Args):
    #Initialize Trainer Class
    checkpoint_path = "./Checkpoints/" + Args.resume if Args.resume is not None else None
    TrManager = Trainer(load_config(Args.config), Args.dataset, eval(Args.warp), eval(Args.trainval), eval(Args.noval), checkpoint_path)
    random.seed(0)

    #Train the Model
    TrManager.train()
	
       
if __name__=='__main__':
    #Launch Training
    main(args)
