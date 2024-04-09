import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np

def test_img(net_g, dataset, args):
    # testing
    correct = 0
    data_loader = DataLoader(dataset, batch_size=128)
    test_loss = 0
    with torch.no_grad():
        net_g.eval()
        for idx, (data, target) in enumerate(data_loader):
            target = target.type(torch.ByteTensor) #加入ByteTensor Ru
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            log_probs = net_g(data)

            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target.squeeze(dim=-1), reduction='sum').item()
            
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            
        test_loss /= len(data_loader.dataset)
        test_acc = 100.00 * correct / len(data_loader.dataset)
        #print("normal_test_acc:", test_acc)
        return test_acc, test_loss

#Ru
def test_img_5net(nets_w, accs_w, dataset, args): #nets_w[5]
    correct = 0
    data_loader = DataLoader(dataset, batch_size=128)
    log_probs = [None for i in range(len(nets_w))] #log_probs: 
    y_preds = [None for i in range(len(nets_w))] #y_preds: 預測的值[0-7]
    with torch.no_grad():
        for i in range(len(nets_w)):
            nets_w[i].eval()
        for idx, (data, target) in enumerate(data_loader):
            target = target.type(torch.ByteTensor) #加入ByteTensor Ru
            y_pred = []
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            
            for i in range(len(nets_w)):
                log_probs[i] = nets_w[i](data)

            # get the index of the max log-probability
            for i in range(len(nets_w)):
                y_preds[i] = log_probs[i].data.max(1, keepdim=True)[1]
            
            #print("y_preds[0]:", y_preds[0])  
            #accs_w: [66.26213836669922, 55.339805603027344, 87.86407470703125, 59.1019401550293, 66.26213836669922]
            # y_preds.shape():  torch.Size([93, 1])
            # list(y_preds[0].shape):  [128, 1], ...., [128, 1], [93, 1]
            
            # check candidate
            # a => 建立np array
            a = np.zeros((list(y_preds[0].shape)[0], list(y_preds[0].shape)[1]), dtype=int)
            for j in range(list(y_preds[0].shape)[0]): # batch_size: 128, 93
                candidate = [] #candidate: (y_pred, acc) #default:(none, 0.0)
                for i in range(len(nets_w)): # nets_w個數
                    if(y_preds[i][j].item() != None):
                        key = 0 # key: {0 => 新的 append, 1 => 加入既有}
                        for k in range(len(candidate)): #若相同
                            if(y_preds[i][j].item() == candidate[k][0]):
                                candidate[k][1] += accs_w[i]
                                key = 1
                                break
                        if(key == 0):
                            candidate.append([y_preds[i][j].item(), accs_w[i]])
                # candidate finish, choose the y_pred with the highest acc
                highest_acc = 0
                highest_label = 0
                for k in range(len(candidate)):
                    if(candidate[k][1] > highest_acc):
                        highest_acc = candidate[k][1]
                        highest_label = candidate[k][0]
                a[j] = highest_label
            y_pred = torch.from_numpy(a).to(args.device) # numpy array a 轉成 tensor b
            #print("y_pred:",y_pred)
        
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            
        test_acc = 100.00 * correct / len(data_loader.dataset)
        #print("5net_test_acc:", test_acc)
        
        return test_acc
        
        '''
            #k = 1
            #y_pred = y_preds[k]
        '''
        '''
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        
        test_loss /= len(data_loader.dataset)
        test_acc = 100.00 * correct / len(data_loader.dataset)
    
        return test_acc, test_loss
        '''