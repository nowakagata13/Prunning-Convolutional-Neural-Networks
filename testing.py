"""Parts of this code were developed by Lee Jun Hyun, available: https://github.com/LeeJunHyun/Image_Segmentation"""

import os
import torch
from torch import optim
from evaluation import *
from network import U_Net
from network_pruned import U_Net_SP
import csv

class Solver_test(object):
	def __init__(self, test_loader, presaved_model_path, channels_file=None):

        # Path
		self.presaved_model_path = presaved_model_path
		self.result_path = './result/'

		# Data loader
		self.test_loader = test_loader

		# Models
		self.unet = None
		self.optimizer = None

		self.criterion = torch.nn.BCELoss()
		self.augmentation_prob = 0.4

		# Hyper-parameters
		self.lr = 0.0002
		self.beta1 = 0.5
		self.beta2 = 0.99

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		if channels_file is not None:
			self.build_model_SP(channels_file)
		else:
			self.build_model()

	def build_model(self):
		"""Builds the model that is of architecture as the intial U-Net, 
		can be used after weight pruning"""

		self.unet = U_Net(img_ch=3,output_ch=1)
			
		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)


	def build_model_SP(self, channels_file):
		"""Builds the model after structured pruning by reading 
		the exact number of channels for each layer based on the text file"""

		self.unet = U_Net_SP(channels_file, img_ch=3,output_ch=1)
			
		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def test(self):
		"""Train encoder, generator and discriminator."""

		if self.presaved_model_path is None:
			print("No test path")
		else:	
			self.unet.load_state_dict(torch.load(self.presaved_model_path, map_location=torch.device(self.device)))
			print('is Successfully Loaded from %s'%(self.presaved_model_path))
			
			self.unet.train(False)
			self.unet.eval()

			acc = 0.	# Accuracy
			SE = 0.		# Sensitivity (Recall)
			SP = 0.		# Specificity
			PC = 0. 	# Precision
			F1 = 0.		# F1 Score
			JS = 0.		# Jaccard Similarity
			DC = 0.		# Dice Coefficient
			length=0
			for i, (images, GT) in enumerate(self.test_loader):

				images = images.to(self.device)
				GT = GT.to(self.device)
				SR = torch.sigmoid(self.unet(images))
				acc += get_accuracy(SR,GT)
				SE += get_sensitivity(SR,GT)
				SP += get_specificity(SR,GT)
				PC += get_precision(SR,GT)
				F1 += get_F1(SR,GT)
				JS += get_JS(SR,GT)
				DC += get_DC(SR,GT)
						
				length += images.size(0)
					
			acc = acc/length
			SE = SE/length
			SP = SP/length
			PC = PC/length
			F1 = F1/length
			JS = JS/length
			DC = DC/length
			
			print("Testing output accuracy: ", acc)

			f = open(os.path.join(self.result_path,'result_FP.csv'), 'a', encoding='utf-8', newline='')
			wr = csv.writer(f)
			wr.writerow([self.presaved_model_path, acc])
			f.close()
			

			
