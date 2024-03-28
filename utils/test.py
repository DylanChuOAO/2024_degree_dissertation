import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def test_img(net_g, dataset, args):
    # testing
    correct = 0
    data_loader = DataLoader(dataset, batch_size=128)
    test_loss = 0
    with torch.no_grad():
        net_g.eval()
        for idx, (data, target) in enumerate(data_loader):
            target = target.type(torch.ByteTensor) #加入ByteTensor Ru
            #查target是什麼
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

        return test_acc, test_loss

#Ru
def test_img_5net(nets_w ,dataset, args): #nets_w[5]
    correct = 0
    data_loader = DataLoader(dataset, batch_size=128)
    with torch.no_grad():
        for i in range(len(nets_w)):
            nets_w[i].eval()
        for idx, (data, target) in enumerate(data_loader):
            target = target.type(torch.ByteTensor) #加入ByteTensor Ru
            if args.gpu != -1:
                data, target = data.to(args.device), target.to(args.device)
            
            log_probs = [None for i in range(len(nets_w))]
            y_preds = [None for i in range(len(nets_w))]
            for i in range(len(nets_w)):
                log_probs[i] = nets_w[i](data)

            # get the index of the max log-probability
            for i in range(len(nets_w)):
                y_preds[i] = log_probs[i].data.max(1, keepdim=True)[1]
                #print("idx: ",idx)
                #print("y_preds: ",y_preds)
            k = 1
            y_pred = y_preds[k]
            
            correct += y_pred.eq(target.data.view_as(y_pred[i])).long().cpu().sum()
            print(y_pred)
        test_acc = 100.00 * correct / len(data_loader.dataset)
        
        return test_acc