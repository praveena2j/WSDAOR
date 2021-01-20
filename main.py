### Importing torch libraries
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.parallel
import os
import torch.nn.functional as F
from torchsummary import summary

import random
### Importing numpy libraries
from numpy import linalg as LA
import numpy as np
import logging
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import sys
import time

### Importing user defined libraries
from utils import parse_configuration
from datasets import create_dataset
from train import train
from val_DA import validate
from test import Test_UNBC
from utils.exp_utils import default_list_train_val_reader
from utils.exp_utils import pearson
from utils import source_estimate_neutral_frames
from utils import target_estimate_neutral_frames

### Importing libraries for viualization
import utils.utils_progress
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# from tensorboardX import SummaryWriter
# from utils.exp_utils import plot_features
# from visdom import Visdom

### Importing libraries for losses
from losses.mmd_loss import MaximumMeanDiscrepancy
# from losses.MeanLoss import MeanLoss
# from losses.center_loss import CenterLoss
from losses.LabelSmoothing import LSR

### Importing model libraries
from models.pytorch_i3d_new import InceptionI3d
from models.I3DWSDDA import I3D_WSDDA
#from models.VGG_inflated import VggFace

args = argparse.ArgumentParser(description='DomainAdaptation')
args.add_argument('-c', '--config', default=None, type=str,
					  help='config file path (default: None)')
args = args.parse_args()
configuration = parse_configuration(args.config)

# global plotter
# plotter = utils.exp_utils.VisdomLinePlotter(env_name='praveen_Plots',port=8051)
# vis = Visdom()

TestError = []
TestAccuracy = []
SEED = configuration['SEED']

ts = time.time()
path = "MILExperiments/"
Logfile_name = path + str(ts) + configuration['Logfilename']

if os.path.isfile(Logfile_name):
	os.remove(Logfile_name)
logging.basicConfig(filename=Logfile_name, level=logging.INFO)

### Using seed for deterministic perfromance
if (SEED == 0):
	torch.backends.cudnn.benchmark = True
else:
	#print("Using SEED")
	#random.seed(SEED)
	#torch.manual_seed(SEED)
	#torch.backends.cudnn.deterministic = True
	#torch.backends.cudnn.benchmark = False
	#np.random.seed(SEED)

		print("Using SEED")
		random.seed(SEED)
		torch.manual_seed(SEED)
		torch.cuda.manual_seed(SEED)
		#torch.cuda.manual_seed_all(SEED)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		np.random.seed(SEED)
		os.environ["PYTHONHASHSEED"] = str(SEED)

x_epoch = []
record = {'train_loss':[], 'train_mae':[], 'val_loss':[], 'val_mae':[], 'train_pcc':[], 'train_icc':[],
		  'sourcepred_loss':[], 'targetpred_loss':[], 'sourcedomain_loss':[], 'targetdomain_loss':[], 'source_pcc':[],'target_pcc':[], 'val_icc':[],
		  'frame_risk':[], 'sequence_risk':[] ,'test_mae':[], 'test_pcc':[], 'test_icc':[], 'MMDloss':[],
		  'accuracy':[], 'accuracy_0':[], 'accuracy_1':[], 'accuracy_2':[], 'accuracy_3':[], 'accuracy_4':[], 'accuracy_5':[]}
fig = plt.figure()
ax0 = fig.add_subplot(321, title="loss")
ax1 = fig.add_subplot(322, title="mae")
ax2 = fig.add_subplot(323, title="pcc")
ax3 = fig.add_subplot(324, title="icc")
#ax4 = fig.add_subplot(325, title="risk")
ax4 = fig.add_subplot(325, title="accuracy")
ax5 = fig.add_subplot(326, title="individuallosses")
#ax6 = fig.add_subplot(327, title="individuallosses")

def draw_curve(epoch, train_loss, train_mae, train_pcc, source_pred_loss, target_pred_loss, source_domain_loss, target_domain_loss, discrep_loss,
			   train_icc, val_loss, val_mae, source_acc, target_acc, val_icc, frame_risk, sequence_risk, test_pcc, test_mae, test_icc, savelearningcurves, cls_accuracy, class_accuracy_0, class_accuracy_1, class_accuracy_2, class_accuracy_3, class_accuracy_4, class_accuracy_5):
	global record
	record['train_loss'].append(train_loss)
	record['train_mae'].append(train_mae)
	record['train_pcc'].append(train_pcc)
	record['train_icc'].append(train_icc)
	record['sourcepred_loss'].append(source_pred_loss)
	record['targetpred_loss'].append(target_pred_loss)
	record['sourcedomain_loss'].append(source_domain_loss)
	record['targetdomain_loss'].append(target_domain_loss)
	record['MMDloss'].append(discrep_loss)

	record['val_loss'].append(val_loss)
	record['val_mae'].append(val_mae)
	record['source_pcc'].append(source_acc)
	record['target_pcc'].append(target_acc)
	record['val_icc'].append(val_icc)
	record['frame_risk'].append(frame_risk)
	record['sequence_risk'].append(sequence_risk)

	record['test_pcc'].append(test_pcc)
	record['test_mae'].append(test_mae)
	record['test_icc'].append(test_icc)

	record['accuracy'].append(cls_accuracy)
	record['accuracy_0'].append(class_accuracy_0)
	record['accuracy_1'].append(class_accuracy_1)
	record['accuracy_2'].append(class_accuracy_2)
	record['accuracy_3'].append(class_accuracy_3)
	record['accuracy_4'].append(class_accuracy_4)
	record['accuracy_5'].append(class_accuracy_5)

	x_epoch.append(epoch)
	ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
	ax0.plot(x_epoch, record['val_loss'], 'ro-', label='val')

	ax1.plot(x_epoch, record['train_mae'], 'bo-', label='train')
	ax1.plot(x_epoch, record['val_mae'], 'ro-', label='val')
	ax1.plot(x_epoch, record['test_mae'], 'go-', label='test')

	ax2.plot(x_epoch, record['train_pcc'], 'bo-', label='train')
	ax2.plot(x_epoch, record['source_pcc'], 'ro-', label='s_val')
	ax2.plot(x_epoch, record['target_pcc'], 'co-', label='t_val')
	ax2.plot(x_epoch, record['test_pcc'], 'go-', label='test')

	ax3.plot(x_epoch, record['train_icc'], 'bo-', label='train')
	ax3.plot(x_epoch, record['val_icc'], 'ro-', label='val')
	ax3.plot(x_epoch, record['test_icc'], 'go-', label='test')

	ax4.plot(x_epoch, record['accuracy'], 'ro-', label='accuracy')
	ax4.plot(x_epoch, record['accuracy_0'], 'go-', label='accuracy_0')
	ax4.plot(x_epoch, record['accuracy_1'], 'bo-', label='accuracy_1')
	ax4.plot(x_epoch, record['accuracy_2'], 'yo-', label='accuracy_2')
	ax4.plot(x_epoch, record['accuracy_3'], 'ko-', label='accuracy_3')
	ax4.plot(x_epoch, record['accuracy_4'], 'co-', label='accuracy_4')
	ax4.plot(x_epoch, record['accuracy_5'], 'mo-', label='accuracy_5')

	#ax4.plot(x_epoch, record['frame_risk'], 'ro-', label='frame')
	#ax4.plot(x_epoch, record['sequence_risk'], 'go-', label='sequence')
	#ax4.plot(x_epoch, record['sourcedomain_loss'], 'bo-', label='source_domain')
	#ax4.plot(x_epoch, record['targetdomain_loss'], 'go-', label='target_domain')

	ax5.plot(x_epoch, record['targetpred_loss'], 'go-', label='target_pred')
	ax5.plot(x_epoch, record['sourcepred_loss'], 'ro-', label='source_pred')
	ax5.plot(x_epoch, record['sourcedomain_loss'], 'bo-', label='source_domain')
	ax5.plot(x_epoch, record['targetdomain_loss'], 'co-', label='target_domain')
	#ax5.plot(x_epoch, record['MMDloss'], 'mo-', label='discrep_loss')

	#ax6.plot(x_epoch, record['sourcepred_loss'], 'ro-', label='source_pred')
	#ax6.plot(x_epoch, record['MMDloss'], 'mo-', label='discrep_loss')

	if epoch == 0:
		ax0.legend()
		ax1.legend()
		ax2.legend()
		ax3.legend()
		ax4.legend()
		ax5.legend()
	fig.savefig(savelearningcurves + ".png")

for i in range(16, 25):
	i = 16	
	#if (i == 16):
	#	continue
	savelearningcurves = configuration['learningcurves_name'] + str(i)
	print("Subject" + str(i))
	logging.info("Subject" + str(i))

	start_epoch = configuration['model_params']['start_epoch']
	total_epoch = configuration['model_params']['max_epochs']
	savemodel_path = configuration['model_params']['export_path']
	savemodel = configuration['model_params']['savedmodelname']
	print_freq = configuration['printout_freq']
	ModeofPred = configuration['ModeofPred']
	Freeze = configuration['model_params']['Freeze']

	# Model Inititalization
	#cnn_lstm_model = VggFace()
	i3d = InceptionI3d(400, in_channels=3)
	i3d.load_state_dict(torch.load('pretrainedweights/rgb_imagenet.pt'))
	cnn_lstm_model = I3D_WSDDA(i3d)
	cnn_lstm_model.cuda()
	cnn_lstm_model = nn.DataParallel(cnn_lstm_model)
	#cnn_lstm_model.load_state_dict(torch.load('savedweights/Inception_FS_SourceLabels_UDA_2LR_acc' + str(i) + '.t7')['net'])
	#cnn_lstm_model.load_state_dict(torch.load('savedweights/Inception_FS_SourceLabels_UDA_2LR_ce_cb_new_t_acc' + str(i) + '.t7')['net'])

	#for name, param in cnn_lstm_model.named_parameters():
	#	if'i3d_WSDDA'in name:
	#		with torch.no_grad():
	#			cnn_lstm_model.module.i3d_WSDDA.logits.conv3d.weight.copy_(state_dict[name])
	#		param.requires_grad =False

	# Loss Function Initialization
	criterion = LSR().cuda()
	# ml_loss = MeanLoss().cuda()
	# criterion_cent = CenterLoss(num_classes=6, feat_dim=512)
	# criterion = nn.SmoothL1Loss().cuda()
	pred_criterion = nn.MSELoss().cuda()
	MMD_criterion = MaximumMeanDiscrepancy().cuda()
	domain_criterion = nn.BCEWithLogitsLoss().cuda()

	# Optimizer Initialization
	# optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=0.5)
	optimizer_domain = torch.optim.SGD(cnn_lstm_model.parameters(), lr=configuration['model_params']['domain_lr'],
									momentum=configuration['model_params']['momentum'],
								weight_decay=configuration['model_params']['weight-decay'])
	optimizer = torch.optim.SGD(cnn_lstm_model.parameters(),
									lr=configuration['model_params']['lr'],
								momentum=configuration['model_params']['momentum'],
								weight_decay=configuration['model_params']['weight-decay'])
	#optimizer = torch.optim.Adam(cnn_lstm_model.parameters(), lr=configuration['model_params']['lr'])

	# Loading Data
	print('==> Preparing data..')
	source_filelist = "train_valence_list_50.txt"
	source_vallist = "test_valence_list_50.txt"
	sourcelabel_path = configuration['source_train_dataset_params']['dataset_labelpath']
	source_train_subjects = configuration['source_train_dataset_params']['Numofsourcetrainsubs']
	source_label_files = [source_filelist, source_vallist]
	source_trainlist, source_vallist = default_list_train_val_reader(sourcelabel_path,
										source_label_files, source_train_subjects, "source")
	logging.info("Loading source train data")
	source_trainloader, source_valloader = create_dataset(
								configuration['source_train_dataset_params'], source_trainlist)
	print('The number of source training batches = {0}'.format(
												len(source_trainloader)))
	logging.info('The number of source training batches = {0}'.format(
												len(source_trainloader)))

	#target_filelist = "../Datasets/UNBC-McMaster/list_full.txt"
	target_label_files = ["list_train.txt"]
	target_train_subjects = configuration['target_train_dataset_params']['Numoftargettrainsubs']
	targetlabel_path = configuration['target_train_dataset_params']['dataset_labelpath']
	target_trainlist, target_vallist = default_list_train_val_reader( targetlabel_path
											+ str(i) + "/", target_label_files, target_train_subjects, "target")

	logging.info("Loading target train data")
	target_trainloader = create_dataset(
								configuration['target_train_dataset_params'], target_trainlist)
	print('The number of target training batches = {0}'.format(
												len(target_trainloader)))
	logging.info('The number of target training batches = {0}'.format(
												len(target_trainloader)))
	#if (configuration['target_train_dataset_params']['ModeofSup'] == 0):
	print("validating on source data")
	logging.info("Loading source val data")
	#source_valloader = create_dataset(
	#							configuration['source_val_dataset_params'], source_vallist)
	print('The number of source validation batches = {0}'.format(
												len(source_valloader)))
	logging.info('The number of source validation batches = {0}'.format(
												len(source_valloader)))
	#else:
	print("validating on target data")
	logging.info("Loading target val data")
	target_valloader = create_dataset(
								configuration['target_val_dataset_params'], target_vallist)
	print('The number of target validation batches = {0}'.format(
												len(target_valloader)))
	logging.info('The number of target validation batches = {0}'.format(
												len(target_valloader)))
	target_testlist = configuration['target_test_dataset_params']['dataset_labelpath'] + str(
		i) + "/" + "list_val.txt"
	logging.info("Loading target test data")
	target_testloader = create_dataset(
								configuration['target_test_dataset_params'], target_testlist)
	print('The number of target test batches = {0}'.format(
												len(target_testloader)))
	logging.info('The number of target test batches = {0}'.format(
												len(target_testloader)))
	#train_mean, train_std = online_mean_and_sd(source_trainloader)
	#train_mean, train_std = online_mean_and_sd(target_trainloader)
	# val_mean, val_std = online_mean_and_sd(trainloader)
	# test_mean, test_val = online_mean_and_sd(trainloader)

	#if (Freeze == 1):
	#	print("Entered in Freezing mode")
		#source_estimate_neutral_frames.estimate_neutral_frames(cnn_lstm_model, source_trainlist, configuration['source_train_dataset_params'], "sourcetrain")
		#source_estimate_neutral_frames.estimate_neutral_frames(cnn_lstm_model, source_vallist, configuration['source_val_dataset_params'], "sourceval")
		#target_estimate_neutral_frames.estimate_neutral_frames(cnn_lstm_model, target_trainlist, configuration['target_train_dataset_params'], "targettrain")

	#if (Freeze == 1):
	#	features = []
	#	for batch_idx, source in enumerate(target_testloader):
	#		with torch.no_grad():
	#			source_inputs, source_labels, _ = source
	#			sourcefeature, source_outputs, source_domain_output = cnn_lstm_model(source_inputs, 0, 0, 0)
	#			t = source_inputs.size(2)
	#			sourceframe_feature = F.interpolate(sourcefeature.squeeze(3).squeeze(3), t, mode='linear')#.squeeze(1)
	#			sourceframe_feature = sourceframe_feature.view(sourceframe_feature.shape[0]*sourceframe_feature.shape[2], -1)#.squeeze()
	#			source_labels = source_labels.view(source_labels.shape[0]*source_labels.shape[1], -1)
	#			features.append(sourceframe_feature.detach().cpu().numpy())
	#	features = np.concatenate(features, 0)
	#	target_test_neutral_feat = np.reshape(np.mean(features, axis=0), (1,1,1024))
	#	np.save('testneutral.npy', target_test_neutral_feat)

	#Test_loss, Test_PCC, Test_MAE, Test_ICC,accuracy, class_accuracy_0, class_accuracy_1, class_accuracy_2, class_accuracy_3, class_accuracy_4, class_accuracy_5= Test_UNBC(target_testloader, cnn_lstm_model, 1, "Subject" + str(i), 1, 0)
	#print("PCC : " + str(Test_PCC))
	#print("ICC : " + str(Test_ICC))
	#print("MAE : " + str(Test_MAE))
	#print("Accuracy : " + str(accuracy))
	#Test_loss, Test_PCC, Test_MAE, Test_ICC, accuracy, class_accuracy_0, class_accuracy_1, class_accuracy_2, class_accuracy_3, class_accuracy_4, class_accuracy_5= Test_UNBC(target_testloader, cnn_lstm_model, 1, "Subject" + str(i), 0, 0)
	#print("PCC : " + str(Test_PCC))
	#print("ICC : " + str(Test_ICC))
	#print("MAE : " + str(Test_MAE))
	#print("Accuracy : " + str(accuracy))
	#sys.exit()

	training_error = []
	validation_error = []
	PCC_test = []
	ICC_test = []
	MAE_test = []
	PCC_s_val = []
	PCC_t_val = []
	ICC_val = []
	MAE_val = []
	MMD = []
	MMD_train = []
	MMAE_test = []
	weighted_fscore_test = []

	test_accuracy = []
	test_class_accuracy_0 = []
	test_class_accuracy_1 = []
	test_class_accuracy_2 = []
	test_class_accuracy_3 = []
	test_class_accuracy_4 = []
	test_class_accuracy_5 = []

	best_Val_frame = 10000000
	best_seq_frame = 10000000
	best_Val_s_acc = 0
	best_Val_t_acc = 0
	best_val_icc = 0
	best_mmae = 1000000
	best_weighted_fscore = 0
	SourceModeofSup = configuration['source_train_dataset_params']['ModeofSup']
	TargetModeofSup = configuration['target_train_dataset_params']['ModeofSup']
	print('==> Training started..')
	for epoch in range(start_epoch, total_epoch):

		# train for one epoch
		print("Training")
		logging.info("Training")

		Training_loss, Training_acc, Training_MAE, Training_ICC, source_features, target_features, source_pred_loss_, target_pred_loss_, source_domain_loss_, target_domain_loss_, discrep_loss = train(source_trainloader, target_trainloader, cnn_lstm_model, pred_criterion, criterion, optimizer_domain, MMD_criterion, optimizer,
											configuration['model_params'], epoch, "Subject" + str(i), print_freq, SourceModeofSup, TargetModeofSup, Freeze)
		print("Validating")
		#logging.info("Validating")
		# evaluate on validation set

		frame_risk, sequence_risk, Valid_loss, source_acc, target_acc, Valid_MAE, Valid_ICC, Valid_decrep_loss = validate(source_features, target_features,
			source_valloader, target_valloader, cnn_lstm_model, pred_criterion, epoch, "Subject" + str(i), TargetModeofSup, Freeze)

		Test_loss, Test_PCC, Test_MAE, Test_MMAE, Test_ICC, accuracy, weighted_fscore, class_accuracy_0, class_accuracy_1, class_accuracy_2, class_accuracy_3, class_accuracy_4, class_accuracy_5= Test_UNBC(target_testloader, cnn_lstm_model, epoch, "Subject" + str(i), ModeofPred, Freeze)

		draw_curve(epoch, Training_loss, Training_MAE, Training_acc, source_pred_loss_, target_pred_loss_, source_domain_loss_, target_domain_loss_, discrep_loss,
				   Training_ICC, Valid_loss, Valid_MAE, source_acc, target_acc, Valid_ICC, frame_risk, sequence_risk, Test_PCC, Test_MAE, Test_ICC, savelearningcurves, accuracy, class_accuracy_0, class_accuracy_1, class_accuracy_2, class_accuracy_3, class_accuracy_4, class_accuracy_5)

		PCC_test.append(Test_PCC)
		ICC_test.append(Test_ICC)
		MAE_test.append(Test_MAE)
		test_accuracy.append(accuracy)
		test_class_accuracy_0.append(class_accuracy_0)
		test_class_accuracy_1.append(class_accuracy_1)
		test_class_accuracy_2.append(class_accuracy_2)
		test_class_accuracy_3.append(class_accuracy_3)
		test_class_accuracy_4.append(class_accuracy_4)
		test_class_accuracy_5.append(class_accuracy_5)
		MMAE_test.append(Test_MMAE)
		weighted_fscore_test.append(weighted_fscore)

		PCC_s_val.append(source_acc)
		PCC_t_val.append(target_acc)
		ICC_val.append(Valid_ICC)
		MAE_val.append(Valid_MAE)
		MMD.append(Valid_decrep_loss)
		MMD_train.append(discrep_loss)

		logging.info('test_accuracy:')
		logging.info(test_accuracy)
		logging.info('PCC_test:')
		logging.info(PCC_test)
		logging.info('ICC_test:')
		logging.info(ICC_test)
		logging.info('MAE_test:')
		logging.info(MAE_test)
		logging.info("MMAE_test")
		logging.info(MMAE_test)
		logging.info("weighted_fscore")
		logging.info(weighted_fscore_test)

		logging.info('PCC_s_val:')
		logging.info(PCC_s_val)
		logging.info('PCC_t_val:')
		logging.info(PCC_t_val)
		logging.info('ICC_val:')
		logging.info(ICC_val)
		logging.info('MAE_test:')
		logging.info(MAE_test)
		logging.info('MMD_Valid:')
		logging.info(MMD)
		logging.info('MMD_train:')
		logging.info(MMD_train)

		if source_acc > best_Val_s_acc:
			#print("Testing")
			#logging.info("Testing")
			print("Test_PCC:" + str(Test_PCC))
			print("Test_ICC:" + str(Test_ICC))
			print("Test_MAE:" + str(Test_MAE))
			print("Test_acc:" + str(accuracy))

			print('Saving..')
			print("best_Val_acc: %0.3f" % source_acc)
			state= {
				'net': cnn_lstm_model.state_dict(),
				'best_Val_pcc': source_acc,
				'best_Val_mae': Valid_MAE,
				'best_Val_icc': Valid_ICC,
				'best_Test_pcc': Test_PCC,
				'best_Test_mae': Test_MAE,
				'best_Test_icc': Test_ICC,
				'best_Val_acc_epoch': epoch,
			}
			if not os.path.isdir(savemodel_path):
				os.mkdir(savemodel_path)
			torch.save(state, os.path.join(savemodel_path, savemodel + '_s_acc' + str(i) + '.t7'))
			best_Val_s_acc = source_acc
			best_Val_acc_epoch = epoch

		if target_acc > best_val_icc:
			#print("Testing")
			#logging.info("Testing")
			print("Test_PCC:" + str(Test_PCC))
			print("Test_ICC:" + str(Test_ICC))
			print("Test_MAE:" + str(Test_MAE))
			print("Test_acc:" + str(accuracy))

			print('Saving..')
			print("best_Val_acc: %0.3f" % target_acc)
			state= {
				'net': cnn_lstm_model.state_dict(),
				'best_Val_pcc': target_acc,
				'best_Val_mae': Valid_MAE,
				'best_Val_icc': Valid_ICC,
				'best_Test_pcc': Test_PCC,
				'best_Test_mae': Test_MAE,
				'best_Test_icc': Test_ICC,
				'best_Val_acc_epoch': epoch,
			}
			if not os.path.isdir(savemodel_path):
				os.mkdir(savemodel_path)
			torch.save(state, os.path.join(savemodel_path, savemodel + '_t_acc' + str(i) + '.t7'))
			best_Val_t_acc = target_acc
			best_Val_acc_epoch = epoch

		if Valid_ICC > best_val_icc:
			#print("Testing")
			#logging.info("Testing")
			print("Test_PCC:" + str(Test_PCC))
			print("Test_ICC:" + str(Test_ICC))
			print("Test_MAE:" + str(Test_MAE))
			print("Test_acc:" + str(accuracy))

			print('Saving..')
			print("best_Val_acc: %0.3f" % Valid_ICC)
			state= {
				'net': cnn_lstm_model.state_dict(),
				'best_Val_pcc': source_acc,
				'best_Val_mae': Valid_MAE,
				'best_Val_icc': Valid_ICC,
				'best_Test_pcc': Test_PCC,
				'best_Test_mae': Test_MAE,
				'best_Test_icc': Test_ICC,
				'best_Val_acc_epoch': epoch,
			}
			if not os.path.isdir(savemodel_path):
				os.mkdir(savemodel_path)
			torch.save(state, os.path.join(savemodel_path, savemodel + '_s_icc' + str(i) + '.t7'))
			best_val_icc = Valid_ICC
			best_Val_acc_epoch = epoch


	print("best_PrivateTest_acc: %0.3f" % target_acc)
	logging.info("best_PrivateTest_acc: %0.3f" % target_acc)
	print("best_PrivateTest_acc_epoch: %d" % best_Val_acc_epoch)
	logging.info("best_PrivateTest_acc_epoch: %d" % best_Val_acc_epoch)
print(TestError)
print(TestAccuracy)
# np.save(path + "/TestError", TestError)
np.save(path + "/TestAccuracy", TestAccuracy)
Final_Accuracy = (sum(TestAccuracy) / len(TestAccuracy))
Final_Error = (sum(TestError) / len(TestError))
print(Final_Error)
print(Final_Accuracy)
logging.info(str(Final_Accuracy))
