import torch.utils.data as data
from PIL import Image, ImageFont, ImageDraw
import os
import os.path
import sys
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision import transforms
import torch
import collections
from scipy import signal
import logging
import cv2
import utils.videotransforms as videotransforms

def default_seq_reader(videoslist, length, stride):
	sequences = []
	num_sequences = 0
	for videos in videoslist:
		video_length = len(videos)
		if (video_length < length):
			continue
		for i in range(0, video_length - length, stride):
			seq = videos[i: i + length]
			if (len(seq) == length):
				sequences.append(seq)
			num_sequences = num_sequences + 1
	return sequences

def default_list_reader(fileList):
	videos = []
	for sub in fileList:
		video_length = 0
		lines = list(sub)
		while (video_length < (len(lines))):
			line = lines[video_length]
			imgPath = line.strip().split(' ')[0]
			find_str = os.path.dirname(imgPath)
			new_video_length = 0
			for line in lines:
				if find_str in line:
					new_video_length = new_video_length + 1
			videos.append(lines[video_length:video_length + new_video_length])
			video_length = video_length + new_video_length
	return videos


class Target_UNBC_ImageList(data.Dataset):
	def __init__(self, root, label_path, fileList, length, flag, stride, list_reader=default_list_reader,
				 seq_reader=default_seq_reader):
		self.data_path = root
		self.label_path = label_path
		self.length = length
		self.stride = stride

		if (flag == 'train'):
			self.videoslist = list_reader(fileList)
			print("Num of target train videos :" + str(len(self.videoslist)))
			self.sequence_list = seq_reader(self.videoslist, self.length, self.stride)
			print("Num of target train sequences :" + str(len(self.sequence_list)))
			logging.info("Num of training videos :" + str(len(self.videoslist)))
			logging.info("Num of target train sequences :" + str(len(self.sequence_list)))
		elif (flag == 'test'):
			videos= []
			logging.info("Loading Test data")
			with open(fileList, 'r') as file:
				lines = list(file)
				videos.append(lines)
			self.videoslist = list_reader(videos)
			#print("Num of target test videos :" + str(len(self.sequence_list)))
			self.sequence_list = seq_reader(self.videoslist, self.length, self.stride)
			print("Num of target test sequences :" + str(len(self.sequence_list)))
			logging.info("Num of test sequences :" + str(len(self.sequence_list)))
		else:
			self.videoslist = list_reader(fileList)
			print("Num of target val videos :" + str(len(self.videoslist)))
			self.sequence_list = seq_reader(self.videoslist, self.length, self.stride)
			print("Num of target val sequences :" + str(len(self.sequence_list)))
			logging.info("Num of target val videos :" + str(len(self.videoslist)))
			logging.info("Num of target val sequences :" + str(len(self.sequence_list)))

		#logging.info("Num of sequences : " + str(len(self.sequence_list)))
		self.flag = flag

	def __getitem__(self, index):
		seq_path = self.sequence_list[index]
		seq, label, subject_id = self.load_data_label(self.data_path, seq_path, self.flag)
		if(self.flag == 'test'):
			label_index = torch.DoubleTensor(label)
		else:
			label_index = torch.as_tensor(label)
		return seq, label_index, subject_id

	def __len__(self):
		return len(self.sequence_list)

	def load_data_label(self, data_path, SeqPath, flag):
		if (flag == 'test'):
			data_transforms = transforms.Compose([videotransforms.CenterCrop(224),
												 #transforms.ToTensor(),
										#transforms.Normalize(mean=[0.318, 0.364, 0.512], std=[0.189, 0.159, 0.150])
	])
			inputs = []
			lab = []
			for data in SeqPath:
				imgPath = data.split(" ")[0]
				label = data.split(" ")[1]
				head_tail = os.path.normpath(imgPath)
				ind_comps = head_tail.split(os.sep)
				subject_id = ind_comps[-3]
				img = cv2.imread(data_path + imgPath)[:, :, [2, 1, 0]]
				w,h,c = img.shape
				if w == 0:
					continue
				else:
					img = cv2.resize(img, (224, 224))[:, :, [2, 1, 0]]

				img = (img/255.)*2 - 1
				#label = (float(label)/5.)*2 - 1
				#img = (img/255.)
				#img = (img - [0.323, 0.366, 0.515])/[0.152, 0.155, 0.176]
				inputs.append(img)
				lab.append(float(label))
			imgs=np.asarray(inputs, dtype=np.float32)
			imgs = data_transforms(imgs)
			return torch.from_numpy(imgs.transpose([3,0,1,2])), lab, subject_id

		else: #if (flag == 'train'):
			data_transforms = transforms.Compose([videotransforms.RandomCrop(224),
										   videotransforms.RandomHorizontalFlip(),
											  #transforms.ToTensor(),
										#transforms.Normalize(mean=[0.318, 0.364, 0.512], std=[0.189, 0.159, 0.150])
	])
			inputs = []
			lab = []
			for data in SeqPath:
				imgPath = data.split(" ")[0]
				label = data.split(" ")[1]
				head_tail = os.path.normpath(imgPath)
				ind_comps = head_tail.split(os.sep)
				subject_id = ind_comps[-3]
				img = cv2.imread(data_path + imgPath)[:, :, [2, 1, 0]]
				w,h,c = img.shape
				if w == 0:
					continue
				else:
					img = cv2.resize(img, (224, 224))[:, :, [2, 1, 0]]

				img = (img/255.)*2 - 1
				#label = (float(label)/5.)*2 - 1
				#img = (img/255.)
				#img = (img - [0.323, 0.366, 0.515])/[0.152, 0.155, 0.176]
				inputs.append(img)
				lab.append(float(label))
			label = max(lab)
			imgs=np.asarray(inputs, dtype=np.float32)
			imgs = data_transforms(imgs)
			return torch.from_numpy(imgs.transpose([3,0,1,2])), label, subject_id

