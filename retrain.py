"""Parts of this code were devloped by Lee Jun Hyun, available: https://github.com/LeeJunHyun/Image_Segmentation"""

import torch
import numpy as np
from torch import optim

from tqdm import tqdm
from network_pruned import U_Net_SP
from evaluation import get_accuracy
from data_loader import get_loader


def retrain(model_prune_path, channels_file, criterion, data_train, epochs=1, batch_size=2, gpu=False, scale=0.5, lr = 0.1):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_prune = U_Net_SP(channels_file, img_ch=3,output_ch=1)
    model_prune.load_state_dict(torch.load(model_prune_path, map_location=device))
    optimizer = optim.Adam(list(model_prune.parameters()), 0.0002, [0.5, 0.99])

    for epoch in range(epochs):

        print("epoch", epoch, "out of", epochs)

        model_prune.train(True)
        epoch_loss = 0
        
        acc = 0.	# Accuracy
        SE = 0.		# Sensitivity (Recall)
        SP = 0.		# Specificity
        PC = 0. 	# Precision
        F1 = 0.		# F1 Score
        JS = 0.		# Jaccard Similarity
        DC = 0.		# Dice Coefficient
        length = 0

        for i, (images, GT) in enumerate(data_train):

            SR = model_prune(images)
            SR_probs = torch.sigmoid(SR)
            SR_flat = SR_probs.view(SR_probs.size(0),-1)

            GT_flat = GT.view(GT.size(0),-1)
            loss = criterion(SR_flat,GT_flat)
            epoch_loss += loss.item()

            # Backprop + optimize
            reset_grad(model_prune)
            loss.backward()
            optimizer.step()

            acc += get_accuracy(SR,GT)
            length += images.size(0)

        acc = acc/length

        # Print the log info
        print('Training accuracy', acc)
    
    print("Finished retraining")
    """
    torch.save(model_prune.state_dict(), "./models/FineTuned.pkl")

    valid_loader = get_loader(image_path='./dataset/valid1',
                        image_size=224,
                        batch_size=1,
                        num_workers=8,
                        mode='valid',
                        augmentation_prob=0.4)

    model_prune.train(False)
    model_prune.eval()

    acc = 0.	# Accuracy
    length=0
    for i, (images, GT) in enumerate(valid_loader):

        images = images.to(device)
        GT = GT.to(device)
        SR = torch.sigmoid(model_prune(images))
        acc += get_accuracy(SR,GT)
        length += images.size(0)
        
    acc = acc/length

    print('[Validation] Acc:', acc)"""

def reset_grad(model):
	"""Zero the gradient buffers."""
	model.zero_grad()
        