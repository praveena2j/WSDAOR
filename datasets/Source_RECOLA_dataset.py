import torch.utils.data as data
from PIL import Image
import os
import os.path
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision import transforms
import torch
from scipy import signal
import utils.videotransforms as videotransforms
import math
import cv2
from utils.exp_utils import online_mean_and_sd
import logging

def default_seq_reader(videoslist, length, stride):
	sequences = []
	maxVal = 0.711746
	minVal = 0.00 #-0.218993
	for videos in videoslist:
		video_length = len(videos)
		if (video_length < length):
			continue
		images = []
		img_labels = []
		for img in videos:
			imgPath, label = img.strip().split(' ')
			img_labels.append(abs(float(label)))
			#img_labels.append(float(label))
			images.append(imgPath)
		medfiltered_labels = signal.medfilt(img_labels, 3)
		#normalized_labels = (medfiltered_labels-minVal)/(maxVal-minVal)
		normalized_labels = (medfiltered_labels-minVal)/(maxVal-minVal)
		normalized_labels = (medfiltered_labels-minVal)/(maxVal-minVal)*5
		vid = list(zip(images, normalized_labels))
		for i in range(0, video_length-length, stride):
			seq = vid[i : i + length]
			if (len(seq) == length):
				sequences.append(seq)
	return sequences

def default_list_reader(label_path, filesList):
	videos = []
	for fileList in filesList:
		video_length = 0
		with open(label_path + fileList, 'r') as file:
			lines = list(file)
			for _ in range(9):
				line = lines[video_length]
				imgPath, _ = line.strip().split(' ')
				find_str = os.path.dirname(imgPath)
				new_video_length = 0
				for line in lines:
					if find_str in line:
						new_video_length = new_video_length + 1
				videos.append(lines[video_length:video_length + new_video_length])
				video_length = video_length + new_video_length
	return videos

class Source_RECOLA_ImageList(data.Dataset):
	def __init__(self, root, label_path, fileList, length, flag, stride, list_reader=default_list_reader, seq_reader = default_seq_reader):
		self.data_path = root
		self.label_path = label_path
		#self.videoslist = list_reader(self.label_path, fileList)
		self.videoslist = fileList
		self.length = length
		self.stride = stride
		self.sequence_list = seq_reader(self.videoslist, self.length, self.stride)
		self.flag = flag
		if (self.flag == 'train'):
			print("Num of source train videos :" + str(len(self.videoslist)))
			print("Num of source train sequences :" + str(len(self.sequence_list)))
			logging.info("Num of source train videos :" + str(len(self.videoslist)))
			logging.info("Num of source train sequences :" + str(len(self.sequence_list)))
		else:
			print("Num of source val videos :" + str(len(self.videoslist)))
			print("Num of source val sequences :" + str(len(self.sequence_list)))
			logging.info("Num of source val videos :" + str(len(self.videoslist)))
			logging.info("Num of source val sequences :" + str(len(self.sequence_list)))
		#if (self.flag == 'train'):
		#    print("computing mean and std for RECOLA")
		#    self.train_mean, self.train_std = online_mean_and_sd(self.sequence_list, root)
		#print(self.train_mean)
		#print(self.train_std)
		#sys.exit()

	def __getitem__(self, index):
		seq_path = self.sequence_list[index]
		seq, label, subject_id = self.load_data_label(self.data_path, seq_path, self.flag)
		label_index = torch.DoubleTensor(label)
		return seq, label_index, subject_id

	def __len__(self):
		return len(self.sequence_list)

	def load_data_label(self, data_path, SeqPath, flag):
		if (flag == 'train'):
			data_transforms = transforms.Compose([videotransforms.RandomCrop(224),
										   videotransforms.RandomHorizontalFlip(),
										   #transforms.ToTensor(),
										#transforms.Normalize(mean=[0.389, 0.277, 0.219], std=[0.189, 0.159, 0.150])
	])
		else:
			data_transforms = transforms.Compose([videotransforms.CenterCrop(224),
												#transforms.ToTensor(),
										#transforms.Normalize(mean=[0.389, 0.277, 0.219], std=[0.189, 0.159, 0.150])
			])
		inputs = []
		lab = []
		for image in SeqPath:
			imgPath = image[0]
			head_tail = os.path.normpath(imgPath)
			ind_comps = head_tail.split(os.sep)
			subject_id = ind_comps[-2]

			label = image[1]
			img = cv2.imread(data_path + imgPath)
			#if (img is not None):
			#	img = cv2.resize(img, (224, 224))[:, :, [2, 1, 0]]
			#else:
			#	continue

			w,h,c = img.shape
			if w == 0:
				continue
			else:
				img = cv2.resize(img, (224, 224))[:, :, [2, 1, 0]]

			img = (img/255.)*2 - 1
			#img = (img/255.)
			#img = (img - [0.387, 0.277, 0.218])/[0.188, 0.159, 0.150]
			#label = (float(label+1)/2.)*5.
			inputs.append(img)
			lab.append(float(label))
		imgs=np.asarray(inputs, dtype=np.float32)
		imgs = data_transforms(imgs)
		return torch.from_numpy(imgs.transpose([3,0,1,2])), lab, subject_id
