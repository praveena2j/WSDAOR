import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.nn.utils import clip_grad_norm_
#from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn.functional as F

from utils.argmax import SoftArgmax1D
import os.path as osp
import os
from numpy import linalg as LA
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import logging
import sys
import pickle

#from losses.mmd_loss import mmd_loss
from utils.exp_utils import plot_features
from utils.exp_utils import plot_features_DA
from utils.exp_utils import Normalize
import utils.utils_progress as utils_progress
import utils.exp_utils as exp_utils
from utils.exp_utils import pearson
from EvaluationMetrics.ICC import compute_icc
from utils.exp_utils import plot_grad_flow

#writer = SummaryWriter('runs/WSLtoUNSL')

def _one_hot(labels, classes, value=1):
		"""
			Convert labels to one hot vectors

		Args:
			labels: torch tensor in format [label1, label2, label3, ...]
			classes: int, number of classes
			value: label value in one hot vector, default to 1

		Returns:
			return one hot format labels in shape [batchsize, classes]
		"""
		one_hot = torch.zeros(labels.size(0), classes)

		#labels and value_added  size must match
		labels = labels.view(labels.size(0), -1)

		value_added = torch.Tensor(labels.size(0), 1).fill_(value)

		value_added = value_added.to(labels.device)
		one_hot = one_hot.to(labels.device)
		one_hot.scatter_add_(1, labels, value_added)

		return one_hot


def train(source_train_loader, target_train_loader, model, pred_criterion, criterion, optimizer_domain, MMD_criterion, optimizer, model_params, epoch, subject, print_freq, SourceModeofSup, TargetModeofSup, freeze):
	print('\nEpoch: %d' % epoch)
	logging.info('\nEpoch: %d' % epoch)

	if (model_params['vis_feat'] > 0):
		print("visualizaing features")
		dirname = "Feat_vis_orig_lr"
		dir_name = "Feat_vis_orig_lr" + "/" +subject
		dname = dir_name + "/" + 'FrameLevel'
		dname2 = dir_name + "/" + 'SeqLevel'

		if not osp.exists(dirname):
			os.mkdir(dirname)
		if not osp.exists(dir_name):
			os.mkdir(dir_name)
		if not osp.exists(dname):
			os.mkdir(dname)
		if not osp.exists(dname2):
			os.mkdir(dname2)

	# switch to train mode
	model.train()

	#source_total = 0
	#target_total = 0
	running_loss = 0
	source_pred_running_loss = 0
	target_pred_running_loss = 0
	source_domain_running_loss = 0
	target_domain_running_loss = 0
	domain_running_loss = 0
	decrep_loss_running_loss = 0
	learning_rate_decay_start = model_params['learning_rate_decay_start']
	learning_rate_decay_every = model_params['learning_rate_decay_every']
	learning_rate_decay_rate = model_params['learning_rate_decay_rate']

	tar, out = [], []
	source_features = []
	target_features = []
	targetvis_features = []
	targetvis_labels = []
	#all_features, all_labels = [], []
	#all_target_labels, all_source_labels = [], []
	len_dataloader = len(source_train_loader)
	n_epoch = 50
	count_batches = 0
	alpha = 10
	beta = 0.75
	p = (epoch + 1) / n_epoch

	#scheduler.step()
	if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
		frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
		decay_factor = learning_rate_decay_rate ** frac
		current_lr = model_params['lr'] * decay_factor
		utils_progress.set_lr(optimizer, current_lr)  # set the decayed rate
	else:
		current_lr = model_params['lr']
	#print('learning_rate: %s' % str(current_lr))
	#current_lr = scheduler.get_lr()
	if(epoch>0):
		domain_current_lr = model_params['domain_lr'] / ((1+ alpha*p)**beta)
	else:
		domain_current_lr = model_params['domain_lr']

	utils_progress.set_lr(optimizer, current_lr)  # set the decayed rate
	utils_progress.set_lr(optimizer_domain, domain_current_lr)  # set the decayed rate
	print('learning_rate: %s' % str(current_lr))
	logging.info('learning_rate: %s' % str(current_lr))

	if (SourceModeofSup == 2):  ### Full Supervision
		print("Training source data in fullsup mode")

	elif (SourceModeofSup == 1):  ### Weak Supervision
		print("Training source data in weaksup mode")

	if (TargetModeofSup == 2):  ### Full Supervision
		print("Training target data in fullsup mode")

	elif (TargetModeofSup == 1):  ### Weak Supervision
		print("Training target data in weaksup mode")

	else:  ## Unsupervision
		print("Training target data in unsup mode")
	#weight = torch.tensor([1.0, 3.2]).cuda()

	for batch_idx, (source, target) in enumerate(zip(source_train_loader, target_train_loader)):

		if (freeze == 1):
			with open('sourcetrain_neutral_frames.pkl', 'rb') as fp:
				source_neutralframe_dictionary = pickle.load(fp)
			with open('targettrain_neutral_frames.pkl', 'rb') as fp:
				target_neutralframe_dictionary = pickle.load(fp)

		p = float(batch_idx + epoch * len_dataloader) / (n_epoch * len_dataloader)
		alpha = 2. / (1. + np.exp(-10 * p)) - 1
		optimizer.zero_grad()
		optimizer_domain.zero_grad()

		#if(batch_idx) == 2:
		#	break

		if (len(source) == 0):
			print("Source Data Exhausted")
			continue

		target_inputs, target_labels, target_ids = target
		source_inputs, source_labels, source_ids = source

		#Numofsourcesamples = source_labels.size(0)
		#Numoftargetsamples = target_labels.size(0)
		source_labels = source_labels.type(torch.FloatTensor).cuda()#.squeeze()
		source_inputs, source_labels = Variable(source_inputs.cuda()), Variable(source_labels)

		if (freeze == 1):
			sourcefeature, source_outputs, source_domain_output = model(source_inputs, alpha, model_params['DA'], source_neutralframe_dictionary, source_ids)
		else:
			sourcefeature, source_outputs, source_domain_output = model(source_inputs, alpha, model_params['DA'], 0, 0, "source")
		if(model_params['DA'] == 2): ## Domain Loss at frame level
			#domain_source_labels = torch.zeros(source_labels.shape[0]*source_labels.shape[1]).type(torch.LongTensor)
			#domain_source_labels_new = torch.zeros(source_labels.shape[0]*source_labels.shape[1], 2)
			#domain_source_labels_new[range(source_labels.shape[0]*source_labels.shape[1]), domain_source_labels]=1
			#source_domain_loss = F.binary_cross_entropy_with_logits(source_domain_output, domain_source_labels_new.cuda())

			domain_source_labels = torch.zeros(source_domain_output.shape[0]).type(torch.LongTensor)
			domain_source_labels_new = torch.zeros(source_domain_output.shape[0], 2)
			domain_source_labels_new[range(source_domain_output.shape[0]), domain_source_labels]=1
			source_domain_loss = F.binary_cross_entropy_with_logits(source_domain_output, domain_source_labels_new.cuda())

		elif(model_params['DA'] == 1): ## Domain Loss at sequence level
			domain_source_labels = torch.zeros(source_labels.shape[0]).type(torch.LongTensor)
			domain_source_labels_new = torch.zeros(source_labels.shape[0], 2)
			domain_source_labels_new[range(source_labels.shape[0]), domain_source_labels]=1
			source_domain_loss = F.binary_cross_entropy_with_logits(source_domain_output, domain_source_labels_new.cuda())
		else:
			source_domain_loss = 0

		if (SourceModeofSup == 2):  ### Full Supervision
			t = source_inputs.size(2)
			source_outputs = F.interpolate(source_outputs.squeeze(3).squeeze(3), t, mode='linear')#.squeeze(1)
			source_outputs = source_outputs.squeeze(1)#.squeeze(2).squeeze(2)

			#source_outputs = source_outputs.squeeze(1).squeeze(2).squeeze(2)
			source_outputs = source_outputs.view(-1, source_outputs.shape[0]*source_outputs.shape[1])#.squeeze()
			source_pred_labels = source_labels.view(-1, source_labels.shape[0]*source_labels.shape[1])#.squeeze()
		elif (SourceModeofSup == 1):  ### Weak Supervision
			source_outputs = source_outputs.squeeze(1).squeeze(2).squeeze(2)
			source_outputs = torch.max(source_outputs, dim=1)[0]
			source_outputs = source_outputs.view(-1, source_outputs.shape[0])#.squeeze()
			source_pred_labels = torch.max(source_labels, dim=1)[0]
			source_pred_labels = source_pred_labels.view(-1, source_pred_labels.shape[0])#.squeeze()
		else:  ### No source labels
			print("No source labels")
			source_pred_loss = 0

		source_pred_loss = model_params['predloss_weight']*pred_criterion(source_outputs, source_pred_labels)

		#source_domain_loss_ = 0.5*domain_criterion(source_domain_output, domain_source_labels_new.cuda())

		target_labels = target_labels.type(torch.LongTensor).cuda()#.squeeze()
		target_inputs, target_labels = Variable(target_inputs.cuda()), Variable(target_labels)

		if (freeze == 1):
			targetfeature, target_outputs, target_domain_output = model(target_inputs, alpha, model_params['DA'], target_neutralframe_dictionary, target_ids)
		else:
			targetfeature, tar_outputs, target_domain_output = model(target_inputs, alpha, model_params['DA'], 0, 0, "target")

		out_ch  = tar_outputs.size(2)
		num_videos = tar_outputs.size(0)
		tar_outputs = tar_outputs.squeeze(3).squeeze(3)

		target_outputs = tar_outputs.permute(0,2,1)
		#target_outputs = tar_outputs.view(tar_outputs.shape[0]*tar_outputs.shape[2], -1)#.squeeze()

		if(model_params['DA'] == 2): ## Domain Loss at frame level
			#domain_target_labels = torch.ones(target_labels.shape[0]*target_labels.shape[1]).type(torch.LongTensor)
			#domain_target_labels_new = torch.zeros(target_labels.shape[0]*target_labels.shape[1], 2)
			#domain_target_labels_new[range(target_labels.shape[0]*target_labels.shape[1]), domain_target_labels]=1
			#target_domain_loss = F.binary_cross_entropy_with_logits(target_domain_output, domain_target_labels_new.cuda())

			domain_target_labels = torch.zeros(target_domain_output.shape[0]).type(torch.LongTensor)
			domain_target_labels_new = torch.zeros(target_domain_output.shape[0], 2)
			domain_target_labels_new[range(target_domain_output.shape[0]), domain_target_labels]=1
			target_domain_loss = F.binary_cross_entropy_with_logits(target_domain_output, domain_target_labels_new.cuda())

		elif(model_params['DA'] == 1): ## Domain Loss at sequence level
			domain_target_labels = torch.ones(target_labels.shape[0]).type(torch.LongTensor)
			domain_target_labels_new = torch.zeros(target_labels.shape[0], 2)
			domain_target_labels_new[range(target_labels.shape[0]), domain_target_labels]=1
			target_domain_loss = F.binary_cross_entropy_with_logits(target_domain_output, domain_target_labels_new.cuda())
		else:
			target_domain_loss = 0

		if (TargetModeofSup == 2):  ### Full Supervision
			#target_outputs = target_outputs.squeeze(2).squeeze(2)
			target_outputs = target_outputs.view(target_outputs.shape[0]*target_outputs.shape[2], -1)#.squeeze()
			target_pred_labels = target_labels.view(target_labels.shape[0]*target_labels.shape[1], -1)#.squeeze()
			target_pred_loss = criterion(target_outputs, target_pred_labels)

			#target_outputs = target_outputs.squeeze(1).squeeze(2).squeeze(2)
			#target_outputs = target_outputs.view(-1, target_outputs.shape[0]*target_outputs.shape[1])#.squeeze()
			#target_pred_labels = target_labels.view(-1, target_labels.shape[0]*target_labels.shape[1])#.squeeze()
			#target_pred_loss = pred_criterion(target_outputs, target_pred_labels)
		elif (TargetModeofSup == 1):  ### Weak Supervision


			#target_outputs = target_outputs.squeeze(1).squeeze(2).squeeze(2)

			softmax_outputs = nn.Softmax(dim=2)
			soft_outputs = softmax_outputs(target_outputs)

			softargmax = torch.argmax(soft_outputs, dim=2)
				
			#print(softargmax)
			indices = torch.max(softargmax, dim = 1)[1]

			#values = torch.max(softargmax, dim = 1)[0]
			#one_hote = torch.zeros(num_videos, out_ch)
			one_hot = torch.zeros(num_videos, out_ch)

			for i in range(num_videos):
				#one_hot[i,:] = (softargmax[i,:] == max(softargmax[i,:])).float()
				one_hot[i,:][indices[i]] =1.0
			sums = torch.sum(one_hot, dim = 1)

			sums = 1.0 / sums.view(num_videos, -1)
			onehot = one_hot * sums
			onehot = onehot.view(onehot.size(0), onehot.size(1), -1).cuda()

			#weigts = _one_hot(indices, 7)
			#weigts = weigts.view(weigts.size(0), weigts.size(1), -1)#.squeeze()
			targ_outputs = soft_outputs * onehot
			targ_outputs = torch.sum(targ_outputs, dim = 1)

			#target_outputs = torch.max(target_outputs, dim=1)[0]#.squeeze()
			#target_outputs = target_outputs.view(-1, target_outputs.shape[0])#.squeeze()
			#target_labels = torch.max(target_labels, dim=1)[0]
			target_labels = target_labels.view(target_labels.shape[0], -1)#.squeeze()
			target_pred_loss = criterion(targ_outputs, target_labels)
			#target_pred_loss = pred_criterion(targ_outputs, target_labels)
		else:  ## Unsupervision
			target_pred_loss = 0

		if (model_params['MMD'] == 2): ### Framelevel features
			t = target_inputs.size(2)
			sourceframe_feature = F.interpolate(sourcefeature.squeeze(3).squeeze(3), t, mode='linear')#.squeeze(1)
			sourceframe_feature = sourceframe_feature.view(sourceframe_feature.shape[0]*sourceframe_feature.shape[2], -1)#.squeeze()
			targetframe_feature = F.interpolate(targetfeature.squeeze(3).squeeze(3), t, mode='linear')#.squeeze(1)
			targetframe_feature = targetframe_feature.view(targetframe_feature.shape[0]*targetframe_feature.shape[2], -1)#.squeeze()
			decrep_loss = 0.01*MMD_criterion(sourceframe_feature, targetframe_feature)
			#decrep_loss = mmd_loss(sourceframe_feature, targetframe_feature)
		elif (model_params['MMD'] == 1): ### Seq level features
			source_feature = torch.max(sourcefeature, dim=2)[0]
			source_feature = source_feature.squeeze(2).squeeze(2)
			target_feature = torch.max(targetfeature, dim=2)[0]
			target_feature = target_feature.squeeze(2).squeeze(2)
			decrep_loss = MMD_criterion(source_feature, target_feature)
			#decrep_loss = mmd_loss(source_feature, target_feature)
		else:
			decrep_loss = 0

		#weight = domain_source_labels_new.shape[0] / domain_target_labels_new.shape[0]
		#domain_labels = torch.cat([domain_source_labels_new, domain_target_labels_new], 0)
		#domain_outputs = torch.cat([source_domain_output, target_domain_output], 0)
		domain_loss = model_params['domainloss_weight']*(source_domain_loss + target_domain_loss)
		loss = source_pred_loss + domain_loss + target_pred_loss

		loss.backward()
		#plot_grad_flow(model.named_parameters())

		#clip_grad_norm_(model.parameters(), 0.1)
		#utils_progress.clip_gradient(optimizer, 0.1)
		optimizer.step()
		optimizer_domain.step()
		#for param in criterion_cent.parameters():
		#    param.grad.data *= (1. / args.weight_cent)
		#optimizer_centloss.step()
		#scheduler.step()

		#source_total += Numofsourcesamples
		#target_total += Numoftargetsamples
		running_loss += loss.item()#*(Numofsourcesamples + Numoftargetsamples)
		source_pred_running_loss += source_pred_loss.item()#*Numofsourcesamples
		target_pred_running_loss += target_pred_loss.item()#*Numofsourcesamples
		if (model_params['DA'] >0):
			source_domain_running_loss += source_domain_loss.item()#*Numofsourcesamples
			target_domain_running_loss += target_domain_loss.item()#*Numoftargetsamples
			domain_running_loss += domain_loss.item()#*Numoftargetsamples
		else:
			source_domain_running_loss = 0
			target_domain_running_loss = 0
			domain_running_loss = 0
		if (model_params['MMD'] >0):
			decrep_loss_running_loss += decrep_loss.item()
		else:
			decrep_loss_running_loss = 0
		count_batches = count_batches + 1

		if (np.isnan(out).any()):
			print(source_inputs)
			print(source_outputs.squeeze().detach().cpu().numpy())
			sys.exit()

		if (model_params['vis_feat'] > 0):
			if (model_params['MMD'] != 1):
				source_feature = torch.max(sourcefeature, dim=2)[0]
				source_feature = source_feature.squeeze(2).squeeze(2)
				target_feature = torch.max(targetfeature, dim=2)[0]
				target_feature = target_feature.squeeze(2).squeeze(2)
			if (model_params['vis_feat'] == 2): ## Frame level features
				t = target_inputs.size(2)
				targetframe_feature = F.interpolate(targetfeature.squeeze(3).squeeze(3), t, mode='linear')#.squeeze(1)
				targetframe_feature = targetframe_feature.view(targetframe_feature.shape[0]*targetframe_feature.shape[2], -1)#.squeeze()
				target_labels = target_labels.view(target_labels.shape[0]*target_labels.shape[1], -1)

			else: #(model_params['vis_feat'] == 1 and model_params['MMD'] == 0) ## Seq level features
				target_labels = torch.max(target_labels, dim=1)[0]
				target_labels = target_labels.view(target_labels.shape[0], -1)

			source_features.append(source_feature.detach().cpu().numpy())
			target_features.append(target_feature.detach().cpu().numpy())
			if(model_params['vis_feat'] == 2):
				targetvis_features.append(targetframe_feature.detach().cpu().numpy())
			else:
				targetvis_features.append(target_feature.detach().cpu().numpy())
			targetvis_labels.append(target_labels.detach().cpu().numpy())

		out = np.concatenate([out, source_outputs.squeeze(0).detach().cpu().numpy()])
		tar = np.concatenate([tar, source_pred_labels.squeeze(0).detach().cpu().numpy()])

		utils_progress.progress_bar(batch_idx, len_dataloader, 'Loss: %.3f | sourcepred_loss: %.3f | targetpred_loss: %.3f |dom_loss: %.3f | decrep_loss: %.3f'
						   % (running_loss, source_pred_running_loss, target_pred_running_loss, domain_running_loss, decrep_loss_running_loss))

	#out, tar = Normalize(out, tar)
	#out, tar = np.asarray(out), np.asarray(tar)
	if (model_params['vis_feat'] > 0):
		source_features = np.concatenate(source_features, 0)
		target_features = np.concatenate(target_features, 0)
		targetvis_features = np.concatenate(targetvis_features, 0)
		targetvis_labels = np.concatenate(targetvis_labels, 0)

		#plot_features_DA(targetframe_features, source_features, target_features, targetframelabels, 6, epoch, dname, prefix='train', subject=subject)
		plot_features_DA(targetvis_features, source_features, target_features, targetvis_labels, 6, epoch, dname2, prefix='train', subject=subject)
		#plot_features(source_features, source_features, all_source_labels, 6, epoch, dname2, prefix='train', subject=subject)

	train_MAE = mean_absolute_error(tar, out)
	#train_MSE = mean_squared_error(tar, out)
	print("MAE :" + str(train_MAE))

	pearson_measure, _ = pearsonr(out, tar)
	print("PCC : " + str(pearson_measure))

	train_icc = compute_icc(out, tar)
	print("ICC : " + str(train_icc))

	#total = source_total + target_total
	source_pred_loss_ = (source_pred_running_loss / count_batches)
	target_pred_loss_ = (target_pred_running_loss / count_batches)

	total_loss = (running_loss / count_batches)
	source_domain_loss_ = (source_domain_running_loss / count_batches)
	target_domain_loss_ = (target_domain_running_loss / count_batches)
	#domain_loss_ = domain_running_loss / count_batches
	descrep_loss_ = decrep_loss_running_loss / count_batches
	return total_loss, pearson_measure, train_MAE, train_icc, source_features, target_features, source_pred_loss_, target_pred_loss_, source_domain_loss_, target_domain_loss_, descrep_loss_
