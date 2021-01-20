import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os.path as osp
import torch.nn.init as init
from EvaluationMetrics.ICC import compute_icc
from torch.autograd import Variable
from utils.exp_utils import Normalize

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

from torchsummary import summary
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F

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


def validate(source_features, target_features,val_loader, model, criterion, epoch, subject, TargetModeofSup):
	# switch to evaluate mode
	model.eval()
	PrivateTest_loss = 0
	running_val_loss = 0
	total = 0

	tar, out = [], []
	seq_tar, seq_out = [], []
	val_MAE = 0
	val_frame_features = []
	val_features = []
	count_source_batches = 0
	count_target_batches = 0
	#all_features, all_labels = [], []

	if (TargetModeofSup == 2):  ## Full Supervision
		print("Validating in fully supervised mode")
	else:    ## Weak Supervision
		print("Validating in weakly supervised mode")

	for batch_idx, (input, target) in enumerate(val_loader):
		#if(batch_idx == 5):
		#	break
		with torch.no_grad():
			inputs = input.cuda()
			inputs = Variable(inputs)
			targets = target.type(torch.FloatTensor).cuda()
			model_targets = Variable(targets)
			model_features, model_outputs, _ = model(inputs, 0, 0, 0)
			## Frame-Level Prediction
			#t = inputs.size(2)
			#model_outputs = F.interpolate(outputs, t, mode='linear')
			#model_outputs = model_outputs.squeeze(1).squeeze(2).squeeze(2)

		#seq_outputs = torch.max(model_outputs, dim=2)[0]
		#seq_targets = torch.max(model_targets, dim=1)[0]

		#seq_outputs = seq_outputs.view(-1, seq_outputs.shape[0])#.squeeze()
		#seq_targets = seq_targets.view(-1, seq_targets.shape[0])#.squeeze()
		count_source_batches = count_source_batches + 1
		if (TargetModeofSup == 2):  ## Full Supervision
			t = inputs.size(2)
			model_outputs = F.interpolate(model_outputs.squeeze(3).squeeze(3), t, mode='linear').squeeze(1)
			model_targets = model_targets.view(-1, model_targets.shape[0]*model_targets.shape[1])
			model_outputs = model_outputs.view(-1, model_targets.shape[0]*model_targets.shape[1])
		else:    ## Weak Supervision
			model_outputs = model_outputs.squeeze(1).squeeze(2).squeeze(2)
			model_outputs = torch.max(model_outputs, dim=1)[0]
			model_targets = torch.max(model_targets, dim=1)[0]

			model_outputs = model_outputs.view(-1, model_outputs.shape[0])#.squeeze()
			model_targets = model_targets.view(-1, model_targets.shape[0])#.squeeze()

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
		out = np.concatenate([out, model_outputs.squeeze(0).detach().cpu().numpy()])
		tar = np.concatenate([tar, model_targets.squeeze(0).detach().cpu().numpy()])

		#seq_out = np.concatenate([seq_out, seq_outputs.squeeze(0).detach().cpu().numpy()])
		#seq_tar = np.concatenate([seq_tar, seq_targets.squeeze(0).detach().cpu().numpy()])

		loss = criterion(model_outputs, model_targets)
		PrivateTest_loss += loss.item()
		total += targets.size(0)
		running_val_loss += loss.item() * targets.size(0)
	PrivateTest_loss = PrivateTest_loss / count_source_batches
	valid_loss = running_val_loss / total

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

	val_MAE = mean_absolute_error(tar, out)
	val_MSE = mean_squared_error(tar, out)
	print("MAE : " + str(val_MAE))
	# print(val_MAE)
	#print("MSE : " + str(val_MSE))
	# print(val_MSE)
	#print("Accuracy : " + str(Val_acc))
	# print(Val_acc)

	pearson_measure, _ = pearsonr(out, tar)
	print("PCC :" + str(pearson_measure))
	# print(pearson_measure)
	#logging.info("Val_accuracy: " + str(pearson_measure))
	#logging.info("MAE : " + str(val_MAE))
	# print((running_val_loss / total), Val_acc)
	val_icc = compute_icc(out, tar)

	print("ICC : " + str(val_icc))
	#logging.info("ICC : " + str(val_icc))
	#plotter.plot('loss', 'val', 'Class Loss', epoch, (running_val_loss / total))
	#plotter.plot('acc', 'val', 'Class Accuracy', epoch, Val_acc.tolist())
	frame_risk = 0
	sequence_risk = 0

	return frame_risk, sequence_risk, PrivateTest_loss, pearson_measure, val_MAE, val_icc

