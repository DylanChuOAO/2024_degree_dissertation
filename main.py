import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

import copy
import numpy as np
import random
from tqdm import trange

from utils.options import args_parser
from utils.sampling import noniid
from utils.dataset import load_data
from utils.test import test_img, test_img_5net
from utils.byzantine_fl import krum, trimmed_mean, fang, dummy_contrastive_aggregation
from utils.attack import compromised_clients, untargeted_attack
from src.aggregation import fedavg
from src.update import BenignUpdate, CompromisedUpdate

#python main.py --gpu 0 --method fedavg --tsboard --c_frac 0.2 --quantity_skew --num_clients 10 --global_ep 5 --debug
#python main.py --gpu 0 --method krum --tsboard --c_frac 0.2 --quantity_skew --num_clients 5 --global_ep 1
#Ru

#Ru
def test(idx, net, test_dataset, args):  
    test_acc, test_loss = test_img(net.to(args.device), test_dataset, args) #Ru
    #print(f"idx: {idx}")
    #print(f"Test accuracy: {test_acc}")
    #print(f"Test loss: {test_loss}")
    
    return test_acc, test_loss
    #test
#Ru end

if __name__ == '__main__':
    # parse args，解析程式的命令行參數。
    args = args_parser()
    # 檢查是否有可用的 GPU，如果有的話，將模型放在 GPU 上進行訓練，否則使用 CPU。
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print("torch.cuda.is_available(): ",torch.cuda.is_available())
    #如果啟用 TensorBoard，則創建一個 TensorBoard 寫入器。
    if args.tsboard:
        writer = SummaryWriter(f'runs/data')

    random.seed(args.seed)
    np.random.seed(args.seed)
    #用來設定隨機初始化的種子，即上述的編號，編號固定，每次取得的隨機數固定。
    torch.manual_seed(args.seed)
    #用於同時設置所有可用的 GPU 的隨機數生成種子。
    torch.cuda.manual_seed_all(args.seed)

    dataset_train, dataset_test, dataset_val = load_data(args)
    
    # early stopping hyperparameters
    cnt = 0
    check_acc = 0

    # sample users (noniid)
    dict_users = noniid(dataset_train, args)
    net_glob = resnet18(num_classes = args.num_classes).to(args.device)
    tmp_net = copy.deepcopy(net_glob).to(args.device)

    net_glob.train()
    
    # copy weights
    w_glob = net_glob.state_dict()

    if args.c_frac > 0:
        compromised_idxs = compromised_clients(args)
    else:
        compromised_idxs = []
        
        
    # 建立一個10x5的二維字典, 每個 clients 儲存 5 個
    # local_storage[][5] = {} 'iter': -1, 'w_idx': -1, 'model': none, 'acc': -1.0, 'loss': -1.0
    local_storage = {i: {j: (-1, -1, None, -1.0, -1.0) for j in range(5)} for i in range(args.num_clients)}
    
    for iter in trange(args.global_ep):
        w_locals = []
        selected_clients = max(int(args.frac * args.num_clients), 1)
        compromised_num = int(args.c_frac * selected_clients)
        idxs_users = np.random.choice(range(args.num_clients), selected_clients, replace=False)

        idxModelTraindata={} # RU: reset the dict, key=idx, value=(model of the idx, trainDataSet of the idx)
        for idx in idxs_users:
            if idx in compromised_idxs:
                # untarget: transmit the fake update
                if args.p == "untarget":
                    w_locals.append(copy.deepcopy(untargeted_attack(net_glob.state_dict(), args)))
                    local = CompromisedUpdate(args = args, dataset = dataset_train, idxs = dict_users[idx])#Ru add
                    idxModelTraindata[idx]=(w_locals[-1], local.data_train_i) #Ru add
                # target: modify the global model’s behavior on a small number of samples
                else: 
                    local = CompromisedUpdate(args = args, dataset = dataset_train, idxs = dict_users[idx])
                    w = local.train(net = copy.deepcopy(net_glob).to(args.device))
                    w_locals.append(copy.deepcopy(w))
                    idxModelTraindata[idx]=(w_locals[-1], local.data_train_i) #Ru add
            else:
                local = BenignUpdate(args = args, dataset = dataset_train, idxs = dict_users[idx])
                w = local.train(net = copy.deepcopy(net_glob).to(args.device))
                w_locals.append(copy.deepcopy(w))
                idxModelTraindata[idx]=(w_locals[-1], local.data_train_i) #Ru add
                # copy weight to net_glob
                #tmp_net.load_state_dict(w)
                #test(idx = idx ,net = tmp_net, test_dataset = dataset_test, args = args) #Ru
                #test(idx = idx ,net = tmp_net, test_dataset = local.data_train_i, args = args) #Ru
                #.ldr_train 帶入.dataset_test #Ru
        # Ru: begin 
        # evaluate each local model in idxModelTraindata={} (key=idx, value=(model of the idx, trainDataSet of the idx))
        # record the result with dict testResults: key=( w_i, t_j)= (acc, loss)
        # e.g. at 4-th iter (iter=4), testing model w_5 (w_i=5) with dataSet T_3 (t_j=3), and 
        # get the testing results: acc=0.6723, loss=1.2345, then: testResults[(4,5,3)]=(0.6723,1.2345)
        testResults = {} # key = (w_i, t_j) = (w_model ,acc, loss)
        
        print("do evaluation for all models, iter=", iter)
        '''
        (w_i testing dataset_train_i)
        for idx in idxs_users:
            (w_i, dataset_i)=idxModelTraindata[idx]
            tmp_net.load_state_dict(w_i)
            test(idx = idx ,net = tmp_net, test_dataset = dataset_i, args = args) #Ru
        '''
        
        for idx_i in idxs_users: #t_i
            for idx_j in idxs_users: #w_j
                if idx_i in compromised_idxs:
                    pass
                else: #Benign Models
                    # copy weight to net_glob
                    tmp_net.load_state_dict(idxModelTraindata[idx_j][0])
                    test_acc, test_loss = test(idx = idx_i ,net = tmp_net, test_dataset = idxModelTraindata[idx_i][1], args = args)
                    testResults[idx_i, idx_j] = (idxModelTraindata[idx_j][0] ,copy.copy(test_acc), copy.copy(test_loss))
                    
        # update local_storage (acc)
        # local_storage[][5] = {} 'iter': -1, 'w_idx': -1, 'model': none, 'acc': -1.0, 'loss': -1.0
        # testResults(w_i, t_j)= (w_model ,acc, loss)
        for idx_i in idxs_users: #t_i
            for idx_j in idxs_users: #w_j
                if idx_i in compromised_idxs:
                    pass
                else: #Benign Models
                    self_check = 0
                    for k in range(5): #跟上回的自己比(w_j)
                        if(local_storage[idx_i][k][1] == idx_j):
                            if(testResults[idx_i, idx_j][1].item() > local_storage[idx_i][k][3]):
                                local_storage[idx_i][k] = (iter, 
                                                           copy.copy(idx_j), 
                                                           copy.copy(testResults[idx_i, idx_j][0]), 
                                                           copy.copy(testResults[idx_i, idx_j][1].item()), 
                                                           copy.copy(testResults[idx_i, idx_j][2]))
                                self_check = 1
                                break
                            else: #(testResults[idx_i, idx_j][0].item() <= local_storage[idx_i][k][3])
                                self_check = 1
                                break
                    if (self_check == 0):
                        tmp_int = 0    
                        tmp_acc = local_storage[idx_i][0][3] 
                        for k in range(5): #找acc最小值
                            for m in range(1, 5): 
                                if(tmp_acc > local_storage[idx_i][m][3]):
                                    tmp_int = m
                                    tmp_acc = local_storage[idx_i][m][3]
                            local_storage[idx_i][tmp_int] = (iter, 
                                                            copy.copy(idx_j), 
                                                            copy.copy(testResults[idx_i, idx_j][0]), 
                                                            copy.copy(testResults[idx_i, idx_j][1].item()), 
                                                            copy.copy(testResults[idx_i, idx_j][2]))
                            break
                        
        #print local_storage_t_i
        #'iter': -1, 'w_idx': -1, 'model': none, 'acc': -1.0, 'loss': -1.0
        for idx_i in range(args.num_clients): #t_i
            if idx_i in compromised_idxs:
                pass
            else:
                '''
                print("idx:",idx_i)
                print(local_storage[idx_i])
                print("\n")
                
                for i in range(5):
                    print("idx_i:", idx_i)
                    print("'iter':", local_storage[idx_i][i][0])
                    print("'w_idx':", local_storage[idx_i][i][1])
                    print("'model':", local_storage[idx_i][i][2])
                    print("'acc':", local_storage[idx_i][i][3])
                    print("'loss':", local_storage[idx_i][i][4])
                    print("\n")
                ''' 
        nets_w = [] 
        #

        for i in range(5): # client[0]中的5個storage
            if(local_storage[0][i][2] != None):
                tmp_net = copy.deepcopy(net_glob).to(args.device)
                tmp_net.load_state_dict(local_storage[0][i][2])
                nets_w.append(tmp_net)
                #tmp_net.load_state_dict(idxModelTraindata[idx_j][0])
        #tests_acc = test_img_5net(nets_w, dataset_test, args)  
        #test_acc, test_loss = test_img(net_glob.to(args.device), dataset_test, args)   
        # Ru: end    
        
        '''
        #Model給我如何執行 (先用w1)
        #測試
        for benignIdx in localBenignModels: #Ru
            #local_DB = modelDB(localBenignModels[benignIdx], benignIdx)
            pass
        #test_img
        
        ##開始加code......
        #w = local.train(net = copy.deepcopy(net_glob).to(args.device))

        '''
        # update global weights
        if args.method == 'fedavg':
            w_glob = fedavg(w_locals)
        elif args.method == 'krum':
            w_glob, _ = krum(w_locals, compromised_num, args)
        elif args.method == 'trimmed_mean':
            w_glob = trimmed_mean(w_locals, compromised_num, args)
        elif args.method == 'fang':
            w_glob = fang(w_locals, dataset_val, compromised_num, args)
        elif args.method == 'dca':
            w_glob = dummy_contrastive_aggregation(w_locals, compromised_num, copy.deepcopy(net_glob), args)
        else:
            exit('Error: unrecognized aggregation technique')

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        test_acc, test_loss = test_img(net_glob.to(args.device), dataset_test, args) #Ru
        #TestData -> (Wg) (原先)
        #Wi_TrainData -> (Wg) (改)
        
        #test_acc, test_loss = test_img(net_glob.to(args.device), dataset_test, args)

        if args.debug:
            print(f"Round: {iter}")
            print(f"Test accuracy: {test_acc}")
            print(f"Test loss: {test_loss}")
            print(f"Check accuracy: {check_acc}")
            print(f"patience: {cnt}")

        if check_acc == 0:
            check_acc = test_acc
        elif test_acc < check_acc + args.delta:
            cnt += 1
        else:
            check_acc = test_acc
            cnt = 0

        # early stopping
        if cnt == args.patience:
            print('Early stopped federated training!')
            break

        # tensorboard
        if args.tsboard:
            writer.add_scalar(f'testacc/{args.method}_{args.p}_cfrac_{args.c_frac}_alpha_{args.alpha}', test_acc, iter)
            writer.add_scalar(f'testloss/{args.method}_{args.p}_cfrac_{args.c_frac}_alpha_{args.alpha}', test_loss, iter)

    if args.tsboard:
        writer.close()
        
