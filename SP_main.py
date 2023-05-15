import torch
import torch.nn as nn
import os.path as osp
from SPclass import SPclass
import copy
from network import U_Net
from data_loader import get_loader
from testing import Solver_test
from retrain import retrain

class Pruning:

    def __init__(self, unet_path, prune_channels):

        self.prune_channels = prune_channels

        self.save_dir = "models/SPruned"
        self.unet_path_p = f'./models\\U_Net-SPruned-{prune_channels}.pkl'
        self.save_txt = f'pruned_channels_{prune_channels}.txt'

        self.data_train = get_loader(image_path='./dataset/train/',
                                image_size=224,
                                batch_size=1, num_workers=8, mode='train',augmentation_prob=0.4)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model Initialization
        self.unet_path = unet_path
        model = U_Net(img_ch=3,output_ch=1)
        model.load_state_dict(torch.load(self.unet_path, map_location=self.device))
        self.model_prune = copy.deepcopy(model)

        self.pruner = SPclass(self.model_prune)  # Pruning handler
        self.criterion = nn.BCELoss()

        self.epoch_loss = 0
        self.taylor_batches = 10 #500
        self.batch_size = 2

        self.ch_dict = self.pruner.ch_dict
        
    def prune(self):

        # Prune & save
        self.pruner.remove_channels(self.prune_channels)
        print('Completed Pruning of %i channels' % self.prune_channels)

        torch.save(self.model_prune.state_dict() , self.unet_path_p)
        print("Pruned model saved as", self.unet_path_p)

        self.pruner.channel_save(self.save_txt)
        print('Pruned structure of network saved to {}...'.format(self.save_txt))


    def test(self):

        test_loader = get_loader(image_path='./dataset/test/',
                                image_size=224,
                                batch_size=1,
                                num_workers=8,
                                mode='test',
                                augmentation_prob=0.4)

        solver_test = Solver_test(test_loader, self.unet_path_p, self.save_txt)
        solver_test.test()

    
    def retrain(self):

        retrain(self.unet_path_p, self.save_txt, self.criterion, self.data_train,
                epochs=1, batch_size=2, gpu=False, scale=0.5, lr=0.1)

        test_loader = get_loader(image_path='./dataset/test/',
                                image_size=224,
                                batch_size=1,
                                num_workers=8,
                                mode='test',
                                augmentation_prob=0.4)

        solver_test = Solver_test(test_loader, "./models/FineTuned.pkl", self.save_txt)
        solver_test.test()


if __name__ == '__main__':
    
    unet_path = './models/U_Net-trained.pkl'
    prune = Pruning(unet_path, 500)
    prune.prune()
    prune.test()
    prune.retrain()

