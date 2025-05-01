# Prunning Convolutional Neural Networks for mobile applications

## Installing the system
To train and test the system, a dataset is required. The full dataset consisting of 2594 images and 12970 corresponding ground truths can be downloaded from https://challenge.isic-archive.com/data/#2018. The downloaded data should be placed in the "dataset" dictionary and divided into the training, validation, and testing sets. The data division can be done manually by creating three folders "test", "valid" and "train", and corresponding "test_GT", "valid_GT", and "train_GT" with their ground truths. The dataset can be also divided into subsets by using the "dataset.py" file included in the model, one needs to parse the arguments as "--origin_data_path", "--origin_GT_path", "--train_ratio", "--valid_ratio", and "--test_ratio", specifying them according to one's data path and needed train-validation-test ratio.

## Dependencies
This system requires Python 3.10 to be installed. Python downloads can be found here -https://www.python.org/All 
Other packages used by the system:torch, torchvision, argparse, os, random, copy, numpy, shutil, csv.
 
## Running the system
From the command line, the system to weight prune the model can be run by "python WeightPruning.py".The model path of the saved pre-trained model is now set to './models/U_Net-trained.pkl' but can be easily changed in the WeightPruning.py file as a constant 'unet_path. The pruning coefficient is now set to 0.5 as it gives the best results based on my evaluation, but it can also be changed in WeightPruning.py file.

To run a structured pruning from the command line, use: "python StructuredPruning.py" where model_path is a path to the pre-trained model and no_of_channeles is a number of channels that we want to remove from the network.

##Space and memory requirements
The space requirement for the pruning system source code is 50 KB.
Space requirements for the trained model used here are 128MB, and an additional 10.8GB for the dataset.

