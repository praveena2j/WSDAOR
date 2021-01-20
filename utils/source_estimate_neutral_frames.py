import pickle
from datasets import create_dataset
import os
import torch.nn.functional as F
import numpy as np
import sys
import torch

def estimate_neutral_frames(cnn_lstm_model, source_trainlist, configuration, flag):
	#subject_neutral_features = []
	source_train_labels = []
	source_train_features = []
	dictionary = {}

	for i in range(len(source_trainlist)):
		trainloader = create_dataset(
								configuration, [source_trainlist[i]])

		videos = source_trainlist[i]
		imgPath, label = videos[0].strip().split(' ')
		head_tail = os.path.normpath(imgPath)
		ind_comps = head_tail.split(os.sep)
		subject_id = ind_comps[-2]
		print(subject_id)

		source_features = []
		labels = []
		for batch_idx, source in enumerate(trainloader):
			with torch.no_grad():
				source_inputs, source_labels, _ = source
				sourcefeature, source_outputs, source_domain_output = cnn_lstm_model(source_inputs, 0, 0, 0)
				t = source_inputs.size(2)
				sourceframe_feature = F.interpolate(sourcefeature.squeeze(3).squeeze(3), t, mode='linear')#.squeeze(1)
				sourceframe_feature = sourceframe_feature.view(sourceframe_feature.shape[0]*sourceframe_feature.shape[2], -1)#.squeeze()
				source_labels = source_labels.view(source_labels.shape[0]*source_labels.shape[1], -1)
				source_features.append(sourceframe_feature.detach().cpu().numpy())
				labels.append(source_labels.detach().cpu().numpy())

		source_features = np.concatenate(source_features, 0)
		labels = np.concatenate(labels, 0)

		mean_face = np.mean(source_features, axis=0)
		source_mc_features = source_features - mean_face
		source_train_features.append(source_mc_features)
		source_train_labels.append(labels)
		print(mean_face.shape)
		#subject_neutral_features.append(mean_face)
		dictionary[subject_id] = mean_face

	#np.save('source_val_features.npy', source_train_features)
	#np.save('source_val_labels.npy', source_train_labels)
	print(dictionary[subject_id])
	f = open(flag + "_neutral_frames.pkl","wb")
	pickle.dump(dictionary,f)
	f.close()

