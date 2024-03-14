import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.test import test_img #Ru
from utils.dataset import DatasetSplit

#Benign: 良性的
class BenignUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        self.local_training_dataset = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True) #Ru
        
    def train(self, net):

        net.train()
        
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                #if(batch_idx<3): print("train: ",batch_idx ,images, labels) #Ru
                labels = labels.type(torch.ByteTensor) #加入ByteTensor Ru
                optimizer.zero_grad()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                log_probs = net(images)
                
                loss = self.loss_func(log_probs, labels.squeeze(dim=-1))
             
                loss.backward()
                
                optimizer.step()

        return net.state_dict() #回傳weight
    def test(self, idx, net, test_dataset, args):
        
        test_acc, test_loss = test_img(net.to(args.device), test_dataset, args) #Ru
        print(f"idx: {idx}")
        print(f"Test accuracy: {test_acc}")
        print(f"Test loss: {test_loss}")
        #test
        
# target(去看paper)
class CompromisedUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, drop_last=True)
        
    def train(self, net):

        
        net_freeze = copy.deepcopy(net)
        
        net.train()
        
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr)
    
        for iter in range(self.args.local_ep):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                optimizer.zero_grad()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                log_probs = net(images)
                
                loss = self.loss_func(log_probs, labels.squeeze(dim=-1))

                loss.backward()
                
                optimizer.step()

        for w, w_t in zip(net_freeze.parameters(), net.parameters()):
            w_t.data = (w_t.data - w.data) * self.args.mp_alpha
        
        return net.state_dict()
#zip()        
"""
>>> a = [ 1 , 2 , 3 ]
>>> b = [ 4 , 5 , 6 ]
>>> zipped = zip ( a , b )     # 回傳一個物件
>>> list ( zipped )  # list() 轉換為列表
[ ( 1 , 4 ) , ( 2 , 5 ) , ( 3 , 6 ) ]
"""