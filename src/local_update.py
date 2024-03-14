from utils.test import test_img

class modelDB(object):
    score = 0
    model = 0 #?
    def __init__(self, benignIdx):
        self.key = benignIdx
    def model_eval(self):
        
        #net_local.load_state_dict(w_glob)
        #test_acc, test_loss = test_img(net_local.to(args.device), dataset_test, args)
        #test_img(Wi, Wi's train dataset, args)