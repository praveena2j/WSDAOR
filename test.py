import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from utils.exp_utils import Normalize

from utils.argmax import SoftArgmax1D
from numpy import linalg as LA
#from tensorboardX import SummaryWriter
from torch.autograd import Variable
import utils.utils_progress
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import utils.exp_utils as exp_utils
from EvaluationMetrics.ICC import compute_icc
from EvaluationMetrics.MMAE import compute_mmae
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from utils.exp_utils import pearson
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F

def Test_UNBC(Testloader, model, epoch, subject, ModeofPred, freeze):
	# switch to evaluate mode
	model.eval()
	test_total = 0
	running_test_loss = 0

	test_tar, test_out = [], []
	# all_features, all_labels = [], []
	if (ModeofPred == 1):
		print("Frame Level Estimation on Test data")
	else:
		print("Sequence Level Estimation on Test data")

	if (freeze == 1):
		diction = {}
		target_test_neutral_feat = np.load('testneutral.npy')
		diction['test'] = target_test_neutral_feat
		print(target_test_neutral_feat.shape)
		subids = ['test']
	correct = 0
	for _, (input, target, _) in enumerate(Testloader):
		with torch.no_grad():
			inputs = input.cuda()
			inputs = Variable(inputs)
			targets = target.type(torch.FloatTensor).cuda()
			model_targets = Variable(targets)
			if (freeze == 1):
				_, model_outputs, _ = model(inputs, 0, 0, diction, subids)
			else:
				_, model_outputs, _ = model(inputs, 0, 0, 0, 0, "target")

			#model_outputs = model_outputs.squeeze(1).squeeze(2).squeeze(2)

			model_outputs = model_outputs.squeeze(3).squeeze(3)

			#print(model_outputs.shape)

			#_, preds = torch.max(model_outputs, 1)
			#print(preds)
			model_outputs = torch.argmax(model_outputs, dim=1)

			#print(model_outputs)
			#sys.exit()
			#print(model_outputs.shape)

			model_outputs = model_outputs.unsqueeze(1)
			#print(model_outputs.shape)

			## Frame-level estimation
			###  Inception
			t = inputs.size(2)
			#values = values.unsqueeze(1).squeeze(3).squeeze(3)
			#model_outputs = model_outputs.squeeze(3).squeeze(3)
			model_outputs = F.interpolate(model_outputs.float(), t, mode='linear')#.squeeze(1)
			#print(model_outputs.shape)
			model_outputs = model_outputs.squeeze(1)
			#print(model_outputs.shape)
			#sys.exit()
			#model_outputs = torch.argmax(model_outputs, dim=1)
			#outputs = model_outputs.view(model_outputs.shape[0]*model_outputs.shape[2], -1)#.squeeze()

			#softargmax = SoftArgmax1D()
			#model_outputs = softargmax(outputs)

			#batchsize = inputs.size(0)
			#model_outputs = model_outputs.view(batchsize, -1)#.squeeze()

			#model_outputs = model_outputs.squeeze(1).squeeze(2).squeeze(2)

			if (ModeofPred == 1):  ## Frame level Estimation
				model_targets = model_targets.view(-1, model_targets.shape[0]*model_targets.shape[1])
				model_outputs = model_outputs.view(-1, model_targets.shape[0]*model_targets.shape[1])

				test_out = np.concatenate([test_out, model_outputs.squeeze().detach().cpu().numpy()])
				test_tar = np.concatenate([test_tar, model_targets.squeeze().detach().cpu().numpy()])

			else:  ## Sequence level Estimation
				model_outputs = torch.max(model_outputs, dim=1)[0]
				model_targets = torch.max(model_targets, dim=1)[0]

				model_outputs = model_outputs.view(-1, model_outputs.shape[0])#.squeeze()
				model_targets = model_targets.view(-1, model_targets.shape[0])#.squeeze()
				test_out = np.concatenate([test_out, np.array([model_outputs.squeeze().detach().cpu().numpy()])])
				test_tar = np.concatenate([test_tar, np.array([model_targets.squeeze().detach().cpu().numpy()])])
		test_total += targets.size(0)

	test_out = test_out.round()

	#conf_mat=confusion_matrix(test_tar, test_out)
	#class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
	accuracy = 100*accuracy_score(test_tar, test_out)
	weighted_fscore = f1_score(test_tar, test_out, average='weighted')

	class_accuracy_0 = 0 #class_accuracy[0]
	class_accuracy_1 = 0 #class_accuracy[1]
	class_accuracy_2 = 0 #class_accuracy[2]
	class_accuracy_3 = 0 #class_accuracy[3]
	class_accuracy_4 = 0 #class_accuracy[4]
	class_accuracy_5 = 0 #class_accuracy[5]


	#print(accuracy)
	#print(class_accuracy)
	#print(test_out)
	#print(test_tar)
	#print(conf_mat)

	#print(test_tar)
	#print(test_out)
	#test_out, test_tar = Normalize(test_out, test_tar)
	#test_out, test_tar = np.asarray(test_out), np.asarray(test_tar)
	#pearson_measure = pearson(test_out, test_tar)
	#all_features = np.concatenate(all_features, 0)
	#all_labels = np.concatenate(all_labels, 0)
	pearson_measure, _ = pearsonr(test_out, test_tar)
	#plot_features(all_features, all_features, all_labels, 6, epoch, dname2, prefix='test', subject=subject)
	test_mae = mean_absolute_error(test_out, test_tar)
	#test_MSE = mean_squared_error(test_tar, test_out)
	print("mae : " + str(test_mae))

	test_icc = compute_icc(test_out, test_tar)
	test_mmae = compute_mmae(test_tar, test_out)
	print("ICC : " + str(test_icc))
	print("MMAE:" + str(test_mmae))
	#logging.info("ICC : " + str(test_icc))

	# print(test_mae)
	#print("mse : " + str(test_MSE))
	# print(test_MSE)
	print("PCC : " + str(pearson_measure))
	#sys.exit()
	#logging.info("Test Accuracy: " + str(pearson_measure))
	#logging.info("MAE : " + str(test_mae))
	return (running_test_loss / test_total), pearson_measure, test_mae, test_mmae, test_icc, accuracy, weighted_fscore, class_accuracy_0, class_accuracy_1, class_accuracy_2, class_accuracy_3, class_accuracy_4, class_accuracy_5
