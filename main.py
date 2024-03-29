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
from utils.test import test_img
from utils.byzantine_fl import krum, trimmed_mean, fang, dummy_contrastive_aggregation
from utils.attack import compromised_clients, untargeted_attack
from src.aggregation import fedavg
from src.update import BenignUpdate, CompromisedUpdate

#from src.local_update import modelDB
#python main.py --gpu 0 --method krum --tsboard --c_frac 0.0 --quantity_skew --num_clients 2 --global_ep 2 
#Ru
if __name__ == '__main__':
    # parse args，解析程式的命令行參數。
    args = args_parser()
    # 檢查是否有可用的 GPU，如果有的話，將模型放在 GPU 上進行訓練，否則使用 CPU。
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
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
    '''
    print("dataset_test: ")
    print(dataset_test)
    '''
    
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
    
    for iter in trange(args.global_ep):
        w_locals = []
        selected_clients = max(int(args.frac * args.num_clients), 1)
        compromised_num = int(args.c_frac * selected_clients)
        idxs_users = np.random.choice(range(args.num_clients), selected_clients, replace=False)

        localBenignModels={} #Ru
        for idx in idxs_users:
            if idx in compromised_idxs:
                # untarget: transmit the fake update
                if args.p == "untarget":
                    w_locals.append(copy.deepcopy(untargeted_attack(net_glob.state_dict(), args)))
                # target: modify the global model’s behavior on a small number of samples
                else: 
                    local = CompromisedUpdate(args = args, dataset = dataset_train, idxs = dict_users[idx])
                    w = local.train(net = copy.deepcopy(net_glob).to(args.device))
                    w_locals.append(copy.deepcopy(w))
            else:
                local = BenignUpdate(args = args, dataset = dataset_train, idxs = dict_users[idx])
                w = local.train(net = copy.deepcopy(net_glob).to(args.device))
                w_locals.append(copy.deepcopy(w))
                # copy weight to net_glob
                tmp_net.load_state_dict(w)
                local.test(idx = idx ,net = tmp_net, test_dataset = dataset_test, args = args) #Ru
                
                #print(localBenignModels[idx]) #Ru
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
        #印出來檢查
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
