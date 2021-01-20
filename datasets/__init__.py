import importlib
from torch.utils import data
import torch
from torch.utils.data import random_split
import numpy as np
import sys
from utils.exp_utils import online_mean_and_sd
import logging
from imbalanceddatasampler.torchsampler.imbalanced import ImbalancedDatasetSampler

def my_collate(batch):
	data = torch.stack([item[0].unsqueeze(0) for item in batch], 0)
	target = torch.Tensor([item[1] for item in batch])
	return [data, target]

def find_dataset_using_name(configuration, labels):
	"""Import the module "data/[dataset_name]_dataset.py".
	In the file, the class called DatasetNameDataset() will
	be instantiated.
	"""
	dataset_name = configuration['dataset_name']

	dataset_filename = "datasets." + dataset_name + "_dataset"
	datasetlib = importlib.import_module(dataset_filename)

	datasetname = dataset_name + '_ImageList'
	datasetclass = getattr(datasetlib, datasetname)

	dataset = datasetclass(root=configuration['dataset_rootpath'], label_path=configuration['dataset_labelpath'],
						fileList=labels, length=configuration['seq_length'], flag = configuration['flag'],
						stride=configuration['stride'])
	return dataset, datasetname

def create_dataset(configuration, label_files):
	"""Create a dataset given the configuration (loaded from the json file).
	This function wraps the class CustomDatasetDataLoader.
		This is the main interface between this package and train.py/validate.py
	"""
	logging.info("Batch Size:" + str(configuration['loader_params']['batch_size']))
	logging.info("Stride:" + str(configuration['stride']))
	dataset, datasetname = find_dataset_using_name(configuration, label_files)
	if (datasetname == 'Target_UNBC_ImageList'):
		if (configuration['flag'] == 'train'):
			targets = []
			for i in range(len(dataset)):
				targets.append(int(dataset[i][1]))
			target = np.asarray(targets)
			numDataPoints = len(targets)
			weights = np.zeros(6, dtype=np.int)
			class_sample_values, class_sample_count = np.unique(target, return_counts=True)

			weights[class_sample_values] = class_sample_count
			weight_new = numDataPoints / (weights + 0.000001)
			samples_weight = torch.from_numpy(weight_new[target])
			sampler = data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
			dataloader = data.DataLoader( dataset, sampler=sampler,
													**configuration['loader_params'])

			#dataloader = data.DataLoader( dataset, #sampler=ImbalancedDatasetSampler(dataset),
			#                                        **configuration['loader_params'])
		else:
			dataloader = data.DataLoader( dataset, **configuration['loader_params'])
		return dataloader
	else:

		train_length = int(0.8*len(dataset)) - 80
		#train_length = int(len(dataset)-80)
		val_length = len(dataset) - train_length
		train_set, val_set = random_split(dataset, [train_length, val_length])
		train_loader = data.DataLoader(train_set, **configuration['loader_params'])
		#train_mean, train_std = online_mean_and_sd(train_loader)
		#print(mean, std)
		validation_loader = data.DataLoader( val_set, batch_size=8, shuffle=False,
												num_workers=4, pin_memory=True)
		#dataloader = data.DataLoader( dataset, **configuration['loader_params'])
		#mean,std = online_mean_and_sd(dataloader)
		#return dataloader
		return train_loader, validation_loader
		#dataloader = data.DataLoader( dataset, **configuration['loader_params'])
	#mean, std = online_mean_and_sd(dataloader)
	#print(mean, std)
	#return dataloader
