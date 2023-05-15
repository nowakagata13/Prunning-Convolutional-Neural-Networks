import torch
import numpy 

class SPclass:

    def __init__(self, model):
        
        self.model = model
        self.convs = []
        self.BatchNorms = []
        self.ch_dict = {}
        self.init_convs()

    def init_convs(self):
        """Initialise modules for pruning"""

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.convs.append(module)
            if isinstance(module, torch.nn.BatchNorm2d):
                self.BatchNorms.append(module)
    
    def rank_channels(self, no_of_channels):
        """Rank channels by their weights"""

        ranking = []
        for conv in self.convs[:-1]:
            for ch in range(conv.weight.shape[0]):
                rank = torch.sum(conv.weight[ch])
                rank = torch.sqrt(((rank)/(conv.weight[ch].nelement()))**2)
                ranking.append(float(rank))

        layers = []
        channels = []
        for i in range(len(self.convs)):
            channels.extend(list(range(self.convs[i].weight.shape[0])))
            layers.extend([i]*self.convs[i].weight.shape[0])

        ranking = torch.Tensor(ranking)
        sorted_rank, sorted_indices = torch.topk(ranking, no_of_channels, largest=False)
        
        sorted_layers = []
        sorted_channels = []
        for i in sorted_indices:
            sorted_layers.append(layers[i])
            sorted_channels.append(channels[i])

        numpy_array = numpy.array(sorted_layers)
        unique, counts = numpy.unique(numpy_array, return_counts=True)
        self.ch_dict = dict(zip(unique, counts))
        print("The channels that will be removed: ", self.ch_dict)
        return sorted_layers, sorted_channels
    
    def create_indices(self):
        """Assign indices for the amount of channels in each convolution using the weight shape"""

        channels = [(list(range(c.weight.shape[1])), list(range(c.weight.shape[0]))) for c in self.convs]
        in_channels, out_channels = list(zip(*channels))
        return in_channels, out_channels
    

    def remove_channels(self, no_of_channels):
        """Remove channels based with the lowest rank"""

        in_channels, out_channels = self.create_indices()
        sorted_layers, sorted_channels = self.rank_channels(no_of_channels)

        for i in range(len(sorted_layers)):
            layer = int(sorted_layers[i]) #index of layer that has a channel
            channel = int(sorted_channels[i])  #index of channel in the layer

            #remove that out channel from the pruned layer
            out_channels[layer].remove(channel)

            #remove in channel from the next layer
            up_conv = True if layer in [10, 13, 16, 19] else False # These tensors are concat with an earlier tensor at bottom.
            if up_conv:
                up_conv_mapping = {10: 7, 13: 5, 16: 3, 19: 1}
                top = self.convs[up_conv_mapping[layer]].weight.shape[0] # number of channels out of the layer in the encoding path
                in_channels[layer + 1].remove(top + channel)  
            else:
                in_channels[layer + 1].remove(channel)

            # if that is a residual layer remove the in channel in the decoding path
            res = True if layer in [1, 3, 5, 7] else False # These tensors are concat at a later conv2d
            if res:
                res_mapping = {1:20, 3:17, 5:14, 7:11}
                in_channels[res_mapping[layer]].remove(channel)

        # Remove parameters using indecies from Convs and BatchNorms
        for i, c in enumerate(self.convs):
            self.convs[i].weight.data = c.weight[out_channels[i], ...][:, in_channels[i], ...]
            self.convs[i].bias.data = c.bias[out_channels[i]]

        for i, bn in enumerate(self.BatchNorms):
            self.BatchNorms[i].weight.data = bn.weight[out_channels[i]]
            self.BatchNorms[i].bias.data = bn.bias[out_channels[i]]
            self.BatchNorms[i].running_mean.data = bn.running_mean[out_channels[i]]
            self.BatchNorms[i].running_var.data = bn.running_var[out_channels[i]]
            
    
    def channel_save(self, path):
        """Save the number of in and out channels after pruning"""
        
        channels = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                channels.append(module.weight.shape[1]) # number of channels in
                channels.append(module.weight.shape[0]) # number of channels out
        
        #save in a file for later use
        with open(path, 'w') as f:
            for item in channels:
                f.write("%s\n" % item)
        