import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os.path as osp
import torch.nn.init as init
from EvaluationMetrics.ICC import compute_icc
from torch.autograd import Variable
from utils.exp_utils import Normalize
import torch.nn as nn
import torch
from utils.argmax import SoftArgmax1D
from numpy import linalg as LA
#from tensorboardX import SummaryWriter
from utils.exp_utils import plot_features

#from visdom import Visdom
import utils.utils_progress
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import utils.exp_utils as exp_utils
import pickle

from torchsummary import summary
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
from losses.MMD_loss import mmd_loss

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

#global plotter
#plotter = exp_utils.VisdomLinePlotter(env_name='praveen_Plots',port=8051)
#logging.basicConfig(filename=path + '/UNBC_source_recola_target_UNBC.log', level=logging.INFO)
#vis = Visdom()

def get_dev_risk(weight, error):
	"""
	:param weight: shape [N, 1], the importance weight for N source samples in the validation set
	:param error: shape [N, 1], the error value for each source sample in the validation set
	(typically 0 for correct classification and 1 for wrong classification)
	"""
	print(weight.shape)
	print(error.shape)

	N, d = weight.shape
	_N, _d = error.shape
	assert N == _N and d == _d, 'dimension mismatch!'
	weighted_error = weight * error
	cov = np.cov(np.concatenate((weighted_error, weight), axis=1),rowvar=False)[0][1]
	var_w = np.var(weight, ddof=1)
	eta = - cov / var_w
	return np.mean(weighted_error) + eta * np.mean(weight) - eta

def get_weight(source_feature, target_feature, validation_feature):
	"""
	:param source_feature: shape [N_tr, d], features from training set
	:param target_feature: shape [N_te, d], features from test set
	:param validation_feature: shape [N_v, d], features from validation set
	:return:
	"""
	N_s, d = source_feature.shape
	N_t, _d = target_feature.shape

	source_feature = source_feature.copy()
	target_feature = target_feature.copy()
	all_feature = np.concatenate((source_feature, target_feature))
	all_label = np.asarray([1] * N_s + [0] * N_t,dtype=np.int32)
	feature_for_train,feature_for_test, label_for_train,label_for_test = train_test_split(all_feature, all_label, train_size = 0.8)

	decays = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
	val_acc = []
	domain_classifiers = []

	for decay in decays:
		domain_classifier = MLPClassifier(hidden_layer_sizes=(d, d, 2),activation='logistic',alpha=decay, random_state = 123, solver='sgd')
		domain_classifier.fit(feature_for_train, label_for_train)
		output = domain_classifier.predict(feature_for_test)
		acc = np.mean((label_for_test == output).astype(np.float32))
		val_acc.append(acc)
		domain_classifiers.append(domain_classifier)
		#print('decay is %s, val acc is %s'%(decay, acc))

	index = val_acc.index(max(val_acc))

	#print('val acc is')
	print(val_acc)

	domain_classifier = domain_classifiers[index]
	print(validation_feature.shape)
	domain_out = domain_classifier.predict_proba(validation_feature)
	return domain_out[:,:1] / domain_out[:,1:] * N_s * 1.0 / N_t


def validate(source_features, target_features,val_loader, target_valloader, model, criterion, epoch, subject, TargetModeofSup, freeze):
	# switch to evaluate mode
	model.eval()
	PrivateTest_loss = 0
	running_val_loss = 0
	descrepancy_loss = 0
	total = 0

	source_tar, source_out = [], []
	target_tar, target_out = [], []

	seq_tar, seq_out = [], []
	val_MAE = 0
	val_frame_features = []
	val_features = []
	count_source_batches = 0
	count_target_batches = 0
	#all_features, all_labels = [], []
	if (freeze == 1):
		with open('sourceval_neutral_frames.pkl', 'rb') as fp:
			neutralframe_dictionary = pickle.load(fp)
	if (TargetModeofSup == 2):  ## Full Supervision
		print("Validating in fully supervised mode")
	else:    ## Weak Supervision
		print("Validating in weakly supervised mode")

	#for batch_idx, (input, target, source_subids) in enumerate(target_valloader):
	for batch_idx, (source, target) in enumerate(zip(val_loader, target_valloader)):
		#it = iter(target_valloader)
		#if(batch_idx == 2):
		#	break

		with torch.no_grad():
			target_inputs, target_labels, target_ids = target
			source_inputs, source_labels, source_ids = source

			source_labels = source_labels.type(torch.FloatTensor).cuda()#.squeeze()
			source_inputs, source_labels = Variable(source_inputs.cuda()), Variable(source_labels)

			target_labels = target_labels.type(torch.FloatTensor).cuda()#.squeeze()
			target_inputs, target_labels = Variable(target_inputs.cuda()), Variable(target_labels)

			if (freeze == 1):
				sourcefeature, model_outputs, _ = model(inputs, 0, 0, neutralframe_dictionary, source_subids)
			else:
				sourcefeature, source_outputs, _ = model(source_inputs, 0, 0, 0, 0, "source")

			## Frame-Level Prediction
			t = source_inputs.size(2)
			source_outputs = F.interpolate(source_outputs.squeeze(3).squeeze(3), t, mode='linear')
			source_outputs = source_outputs.squeeze(1)

			#print(source_outputs.shape)
			#print(source_labels.shape)

			source_labels = source_labels.view(-1, source_labels.shape[0]*source_labels.shape[1])
			source_outputs = source_outputs.view(-1, source_outputs.shape[0]*source_outputs.shape[1])

			#print(source_labels.shape)
			#print(source_outputs.shape)
			_, target_outputs, _ = model(target_inputs, 0, 0, 0, 0, "target")
			#print(target_outputs.shape)
			target_outputs = target_outputs.squeeze(3).squeeze(3)
			target_outputs = target_outputs.permute(0,2,1)
			softmax_outputs = nn.Softmax(dim=2)
			soft_outputs = softmax_outputs(target_outputs)
			softargmax = torch.argmax(soft_outputs, dim=2)
			#print(softargmax.shape)

			#print(model_outputs.shape)
			#model_outputs = model_outputs.squeeze(1).squeeze(2).squeeze(2)
			#if (len(it) > 0):
			#	target_input, target_labels = next(it)
			#	targetfeature, _, target_domain_output = model(target_input, 0, 0, 0)
			#	t = inputs.size(2)
			#	sourceframe_feature = F.interpolate(sourcefeature.squeeze(3).squeeze(3), t, mode='linear')#.squeeze(1)
			##	targetframe_feature = F.interpolate(targetfeature.squeeze(3).squeeze(3), t, mode='linear')#.squeeze(1)
			#	targetframe_feature = targetframe_feature.view(targetframe_feature.shape[0]*targetframe_feature.shape[2], -1)#.squeeze()
			#	decrep_loss = mmd_loss(sourceframe_feature, targetframe_feature)
			#	count_target_batches = count_target_batches + 1
		#seq_outputs = torch.max(model_outputs, dim=2)[0]
		#seq_targets = torch.max(model_targets, dim=1)[0]

		#model_outputs = model_outputs.squeeze(3).squeeze(3)

		#print(model_outputs)
		#model_outputs = model_outputs.permute(0,2,1)
		#_, preds = torch.max(model_outputs, 1)
		#model_outputs = torch.argmax(model_outputs, dim=1)
		#model_outputs = torch.argmax(outputs, dim=1)

		#t = target_inputs.size(2)
		#outputs = F.interpolate(softargmax.unsqueeze(1).float(), t, mode='linear')#.squeeze(1)
		#outputs = F.interpolate(model_outputs.squeeze(3).squeeze(3), t, mode='linear')#.squeeze(1)
		#model_outputs = outputs.squeeze(1)
		#outputs = torch.argmax(outputs, dim=1)

		#outputs = outputs.view(outputs.shape[0]*outputs.shape[2], -1)#.squeeze()

		#softargmax = SoftArgmax1D()
		#model_outputs = softargmax(outputs)

		#batchsize = inputs.size(0)
		#model_outputs = model_outputs.view(batchsize, -1)#.squeeze()

		#seq_outputs = seq_outputs.view(-1, seq_outputs.shape[0])#.squeeze()
		#seq_targets = seq_targets.view(-1, seq_targets.shape[0])#.squeeze()
		count_source_batches = count_source_batches + 1
		#if (TargetModeofSup == 2):  ## Full Supervision
			#model_outputs = model_outputs.squeeze(1).squeeze(2).squeeze(2)
		#	model_targets = target_labels.view(-1, target_labels.shape[0]*target_labels.shape[1])
		#	model_outputs = model_outputs.view(-1, target_labels.shape[0]*target_labels.shape[1])
			
		#else:    ## Weak Supervision
			#model_outputs = model_outputs.squeeze(1).squeeze(2).squeeze(2)
			#model_targets = target_labels.view(-1, model_targets.shape[0])#.squeeze()
		outputs = torch.max(softargmax, dim=1)[0]
		#print(outputs.shape)
		model_outputs = outputs.view(-1, outputs.shape[0])#.squeeze()
		#print(model_outputs.shape)
		#target_labels = torch.max(target_labels, dim=1)[0]
		#print(target_labels.shape)
		model_targets = target_labels.view(-1, target_labels.shape[0])#.
		#print(model_targets.shape)

			#print(model_outputs.shape)
			##print(model_targets.shape)
			#sys.exit()

			#model_outputs = model_outputs.view(-1, model_outputs.shape[0])#.squeeze()
			#model_targets = model_targets.view(-1, model_targets.shape[0])#.squeeze()

		#validation_frame_feature = F.interpolate(model_features.squeeze(3).squeeze(3), t, mode='linear')
		#validation_frame_feature = validation_frame_feature.view(validation_frame_feature.shape[0]*validation_frame_feature.shape[2], -1)

		#validation_feature = torch.max(model_features, dim=2)[0]
		#validation_feature = validation_feature.squeeze(2).squeeze(2)
		#all_features.append(features.data.cpu().numpy())
		#all_labels.append(targets.data.cpu().numpy())

		#val_features.append(validation_feature.detach().cpu().numpy())
		#val_frame_features.append(validation_frame_feature.detach().cpu().numpy())

		#out.append(outputs.data.cpu().numpy())
		#tar.append(targets.data.cpu().numpy())
		source_out = np.concatenate([source_out, source_outputs.squeeze(0).detach().cpu().numpy()])
		source_tar = np.concatenate([source_tar, source_labels.squeeze(0).detach().cpu().numpy()])

		target_out = np.concatenate([target_out, model_outputs.squeeze(0).detach().cpu().numpy()])
		target_tar = np.concatenate([target_tar, model_targets.squeeze(0).detach().cpu().numpy()])

		#seq_out = np.concatenate([seq_out, seq_outputs.squeeze(0).detach().cpu().numpy()])
		#seq_tar = np.concatenate([seq_tar, seq_targets.squeeze(0).detach().cpu().numpy()])
		loss = 0
		#loss = criterion(model_outputs, model_targets)
		PrivateTest_loss = 0 #+= loss.item()
		#descrepancy_loss += decrep_loss
		#total += targets.size(0)
		running_val_loss = 0 #+= loss.item() * targets.size(0)
	#source_out = source_out.round()
	target_out = target_out.round()
	PrivateTest_loss = PrivateTest_loss / count_source_batches
	#descrepancy_loss = descrepancy_loss / count_target_batches
	valid_loss = 0 #running_val_loss / total

	#val_features = np.concatenate(val_features, 0)
	#val_frame_features = np.concatenate(val_frame_features, 0)

	#out, tar = Normalize(out, tar)
	#out, tar = np.asarray(out), np.asarray(tar)

	#frame_error  = ((out - tar)**2)
	#seq_error  = ((seq_out - seq_tar)**2)
	#frame_error = frame_error.reshape(frame_error.shape[0],1)
	#seq_error = seq_error.reshape(seq_error.shape[0],1)

	#weights = get_weight(source_features, target_features, val_features)
	#frame_weights = get_weight(source_features, target_features, val_frame_features)

	#frame_risk = get_dev_risk(frame_weights, frame_error)
	#sequence_risk = get_dev_risk(weights, seq_error)

	val_MAE = mean_absolute_error(source_tar, source_out)
	target_MAE = mean_absolute_error(target_tar, target_out)
	#val_MSE = mean_squared_error(tar, out)
	print("source_MAE : " + str(val_MAE))
	print("target_MAE : " + str(target_MAE))
	# print(val_MAE)
	#print("MSE : " + str(val_MSE))
	# print(val_MSE)
	#print("Accuracy : " + str(Val_acc))
	# print(Val_acc)

	source_pcc, _ = pearsonr(source_out, source_tar)
	target_pcc, _ = pearsonr(target_out, target_tar)
	print("source_PCC :" + str(source_pcc))
	print("target_PCC :" + str(target_pcc))
	# print(pearson_measure)
	#logging.info("Val_accuracy: " + str(pearson_measure))
	#logging.info("MAE : " + str(val_MAE))
	# print((running_val_loss / total), Val_acc)
	val_icc = compute_icc(source_out, source_tar)
	target_icc = compute_icc(target_out, target_tar)

	print("source_ICC : " + str(val_icc))
	print("target_ICC : " + str(target_icc))
	
	#logging.info("ICC : " + str(val_icc))
	#plotter.plot('loss', 'val', 'Class Loss', epoch, (running_val_loss / total))
	#plotter.plot('acc', 'val', 'Class Accuracy', epoch, Val_acc.tolist())
	frame_risk = 0
	sequence_risk = 0
	descrepancy_loss = 0
	return frame_risk, sequence_risk, PrivateTest_loss, source_pcc, target_pcc, val_MAE, val_icc, descrepancy_loss

