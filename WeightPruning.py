import torch
import copy
import torch.nn.utils.prune as prune
from network import U_Net
from testing import Solver_test
from data_loader import get_loader


class WeightPruner:

    def __init__(self, model_path):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = U_Net(img_ch=3,output_ch=1)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device)) #can be changed depending on the device
        self.model_prune = copy.deepcopy(self.model)


    def choose_layers(self):
        """Returns the list of Convolutional and BatchNorm layers that should be pruned"""

        modules_prune = []
        for module in self.model_prune.modules():
            if isinstance(module, torch.nn.Conv2d):
                modules_prune.append(module)
            if isinstance(module, torch.nn.BatchNorm2d):
                modules_prune.append(module)
        return modules_prune
    

    def prune(self, i):
        """Removes i% of weights in the model by setting the mask and applying it to the model"""

        parameters_to_prune = [
            (module, "weight") for module in self.choose_layers()
            ]

        #prune using L1Unstructured method
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=i,
        )

    def apply_pruning(self):
        """Removes the buffer and apply the masks to the original model's weights"""

        for module in self.choose_layers():
            prune.remove(module, "weight")


    def calculate_sparsity(self):
        """Caluclate the number of weights that are zero"""

        modules_pruned = [
            module for module in filter(lambda m: type(m) == torch.nn.Conv2d or type(m) == torch.nn.BatchNorm2d, self.model_prune.modules())
        ]
        zero_weight_n = 0 #number of the weights that are zero
        total_weight_n = 0 #number of all weights in 

        for i in range(len(modules_pruned)):
            zero_weight_n += (torch.sum(modules_pruned[i].weight == 0))
            total_weight_n += modules_pruned[i].weight.nelement()
            module_sparsity = (torch.sum(modules_pruned[i].weight == 0))/(modules_pruned[i].weight.nelement())
            print("The sparsity of module", i, "is {:.2f}%".format(100*module_sparsity))

        print("The total sparsity is {:.2f}%".format(100*zero_weight_n/total_weight_n))


    def test(self, pruned_path):
        "Test the model after pruning"
        
        test_loader = get_loader(image_path='./dataset/test_new/',
                                    image_size=224,
                                    batch_size=1,
                                    num_workers=8,
                                    mode='test',
                                    augmentation_prob=0.4)

        solver_test = Solver_test(test_loader, pruned_path)

        solver_test.test()

    def save(self, i):
        """Saves the pruned model as a state dictionary"""
    
        unet_path_p = './models\\U_Net_WPpruned_{}.pkl'.format(i)
        torch.save(self.model_prune.state_dict() ,unet_path_p)
        print("Pruned model saved in", unet_path_p)

        return unet_path_p


if __name__ == '__main__':
    
    unet_path = './models/U_Net-trained.pkl'
    pruner = WeightPruner(unet_path)
    pruner.prune(0.5)
    pruner.apply_pruning()
    saved_model = pruner.save(0.5)
    pruner.test(saved_model)
