from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.create_episodes import make_episodes
from PIL import Image
import pickle

class loadDataset():
    def __init__(self, txt_file, classes_path, image_dir, N, K, train_set):
        '''
        annotations.pkl and image_paths.pkl were created using following code and pickled to save time
        currently located in ./utils directory
        '''
        # self.txt_file= txt_file
        # DELIMITER = ' '
        # data = []
        # # path= os.path.join(root_dir, txt_file)
        # with open(str(txt_file)) as fr:
        #   for line in fr:
        #     first_col = line.split('{}'.format(DELIMITER))[1]
        #     data.append(first_col.strip())
        # self.annotations = data  # <-- annotations.pkl
        
        # with open(classes_path, 'r') as fp:
        #     lines= [line.rstrip() for line in fp]
        # imagenames=[]
        # for i in range(200):
        #     path= image_dir+lines[i]
        #     files= os.listdir(path)
        #     for j in os.scandir(path):
        #         imagenames.append(j.path)
        # self.image_paths = imagenames  # <-- image_paths.pkl
        
        
        with open('./drive/MyDrive/CV_Project/Code/utils/annotations.pkl','rb') as f:
          self.annotations = pickle.load(f)
        with open('./drive/MyDrive/CV_Project/Code/utils/image_paths.pkl','rb') as f:
          self.image_paths = pickle.load(f)
          
        self.train_df, self.test_df, self.val_df = make_episodes(self.annotations, N, K)
        # same transformation for val and test --> test
        self.curr_set = "train" if train_set == 'train' else "test"

        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        
        
        print("Img path",len(self.image_paths), len(self.annotations))
        self.transform = data_transforms[self.curr_set]
    def getImage(self, idx):
        # img = plt.imread(self.image_paths[idx])
        # print(len(self.image_paths),idx)
        img = Image.open(self.image_paths[idx]).convert('RGB')
        # if img.shape[0]==1:
        #   img = torch.stack([img]*3)
        img = self.transform(img)
        return img

class EpiDataset(Dataset):
    def __init__(self, N, K, txt_file, classes_path, image_dir, data_set):
        self.dataset = loadDataset(txt_file, classes_path, image_dir, N, K, data_set)
        self.df = None
        if data_set == 'train':
            self.df = self.dataset.train_df
        elif data_set == 'test':
            self.df = self.dataset.test_df
        else:
            self.df = self.dataset.val_df
        print(self.df)
        self.N = N
        self.K = K
        # self.preprocess = preprocess()

        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        episode = self.df.loc[idx, :]
        return self.process_episode(episode)


    def process_episode(self, ep):
        supp_set = ep["support_set"].split("@")
        supp_label = ep["support_label"].split("@")
        q_set = ep["query_set"].split("@")
        q_label = ep["query_label"].split("@")

        bin_supp_label, label_vec = self.transform_support_labels(supp_label)
        bin_q_label = self.transform_query_labels(q_label, label_vec, q_set)

        supp_img = self.getImageTensor(supp_set)
        q_img = self.getImageTensor(q_set)

        # sup_idx = torch.randperm(supp_img.size()[0])
        # supp_img = supp_img[sup_idx]
        # bin_supp_label = bin_supp_label[sup_idx]

        # q_idx = torch.randperm(q_img.size()[0])
        # q_img = q_img[q_idx]
        # bin_q_label = bin_q_label[q_idx]

        return {
            "supp_set": supp_img,
            "supp_label": bin_supp_label,
            "q_set": q_img,
            "q_label": bin_q_label
        }


    def getImageTensor(self, img_idx_grp):
        imgTensor = torch.tensor([])
        for img_idx_str in img_idx_grp:
            img_idx = img_idx_str.split(" ")
            imgTensor = torch.cat(( imgTensor, torch.stack([self.dataset.getImage(int(x)) for x in img_idx]) ), dim=0 )
        return imgTensor


    def transform_support_labels(self, labels, one_hot=False):
        eye = torch.eye(self.N, dtype=torch.long)
        label_vec = {}
        if one_hot:
            one_hot_label = torch.tensor([])
            for i in range(self.N):
                one_hot_label = torch.cat((one_hot_label, torch.stack([eye[:,i]]*self.K)),dim=0)
                label_vec[labels[i]] = eye[:,i]
            return one_hot_label, label_vec
        
        label_enc = torch.arange(self.N, dtype=torch.long)
        for i in label_enc:
            label_vec[labels[i]] = i
        return label_enc.repeat_interleave(self.K), label_vec

        

    def transform_query_labels(self, q_label, label_vec, q_set):
        one_hot_label = torch.tensor([], dtype=torch.long)
        for i,grp in enumerate(q_set):
            n = len(grp.split(" "))
            one_hot = label_vec[q_label[i]]
            one_hot_label = torch.cat((one_hot_label, torch.stack([one_hot]*n)),dim=0)
        return one_hot_label




        




        

