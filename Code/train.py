from random import random
from re import T
from nets.model import FSL
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch

from dataloader import *
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.metrics import f1_score

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses

import os,sys
import argparse
import numpy as np

device  = 'cuda'


def euclidean_dist(x, y):
  """
  Computes euclidean distance btw x and y
  Args:
      x (torch.Tensor): shape (n, d). n usually n_way*n_query
      y (torch.Tensor): shape (m, d). m usually n_way
  Returns:
      torch.Tensor: shape(n, m). For each query, the distances to each centroid
  """
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)

def train(model, train_loader, optimizer, epoch, N_way=5, k_shot=5, log_interval=5, fine_tune=False, log_path=None):
    '''
    Train model
    '''
    model.train()
    MSE = nn.MSELoss()
    CE = nn.CrossEntropyLoss()
    CL = losses.ContrastiveLoss()

    
    running_mse = 0.0
    running_cl = 0.0
    running_proto = 0.0
    running_acc = 0.0
    for epi_id, episode in enumerate(train_loader):
        support, support_label, query, query_label = episode['supp_set'].squeeze().to(device), episode['supp_label'].squeeze().to(device), episode['q_set'].squeeze().to(device), episode['q_label'].squeeze().to(device)

        
        optimizer.zero_grad()
        self_attn_feat, img_feat, reconstructed_feat, logits = model(torch.cat((support,query), dim=0))

        z_proto = logits[:N_way*k_shot].view(N_way, k_shot, -1).mean(1)
        z_query = logits[N_way*k_shot:]
        dists = euclidean_dist(z_query, z_proto)

        query_label = Variable(query_label, requires_grad=False)

        
        loss_proto = CE(-dists,query_label)
        y_hat = torch.argmax(F.softmax(-dists, dim=-1), dim=-1)
        acc_val = torch.eq(y_hat, query_label.squeeze()).float().mean()

        loss_cl = CL(self_attn_feat[:N_way*k_shot], support_label)
        loss_mse = MSE(reconstructed_feat, img_feat)
        
        loss =   loss_mse + loss_proto + loss_cl
        

        loss.backward()
        optimizer.step()

        running_mse += loss_mse.item()
        running_cl += loss_cl.item()
        running_proto += loss_proto.item()
        running_acc += acc_val.item()
        if epi_id % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAcc: {:.3f}\tLoss: {:.3f}\tLoss: {:.3f}\tLoss: {:.3f}'.format(
                epoch, epi_id, len(train_loader.dataset),
                       100. * epi_id / len(train_loader), acc_val.item(), loss_mse.item(),  loss_proto.item(), loss_cl.item()))
    print("Train Epoch {}\tAvg. Acc:{:.4f}\tMSE:{:.3f}\tProto:{:.3f}\tCL:{:.3f}\n".format(epoch,running_acc/len(train_loader.dataset),running_mse/len(train_loader.dataset), running_proto/len(train_loader.dataset), running_cl/len(train_loader.dataset)))
    
    if log_path:
        with open(log_path, 'a') as f:
          f.write("{:03d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
            epoch,
            running_acc/len(train_loader.dataset),
            running_mse/len(train_loader.dataset), 
            running_proto/len(train_loader.dataset), 
            running_cl/len(train_loader.dataset)
            )
          )


def evalModel(model, val_loader, epoch, N_way=5, k_shot=5, log_interval=5, fine_tune=False, log_path=None):
    model.eval()
    running_macro = 0.0
    running_micro = 0.0
    running_wted = 0.0
    running_acc = 0.0
    with torch.no_grad():
      for epi_id, episode in enumerate(val_loader):
          support, support_label, query, query_label = episode['supp_set'].squeeze().to(device), episode['supp_label'].squeeze().to(device), episode['q_set'].squeeze().to(device), episode['q_label'].squeeze().to(device)
          
          self_attn_feat, img_feat, reconstructed_feat, logits = model(torch.cat((support,query), dim=0))

          z_proto = logits[:N_way*k_shot].view(N_way, k_shot, -1).mean(1)
          z_query = logits[N_way*k_shot:]

          dists = euclidean_dist(z_query, z_proto)

          y_hat = torch.argmax(F.softmax(-dists, dim=-1), dim=-1)

          y_truth = query_label.squeeze()
          acc_val = torch.eq(y_hat, y_truth).float().mean()

          y_truth = y_truth.cpu().detach().numpy()
          y_hat = y_hat.cpu().detach().numpy()

          macro_F1 = f1_score(y_truth, y_hat, average='macro')
          micro_F1 = f1_score(y_truth, y_hat, average='micro')
          wgted_F1 = f1_score(y_truth, y_hat, average='weighted')

          running_acc += acc_val.item()
          running_macro += macro_F1
          running_micro += micro_F1
          running_wted += wgted_F1

          print("Episode: {}\tAcc: {:.4f}\tMacro-F: {:.3f}\tMicro-F: {:.3f}\tWted: {:.3f}".format(epi_id+1,acc_val.item(), macro_F1, micro_F1, wgted_F1))
    metrics = [
      epoch, 
      running_acc/len(val_loader.dataset), 
      running_macro/len(val_loader.dataset),
      running_micro/len(val_loader.dataset),
      running_wted/len(val_loader.dataset)
    ]
    print("\nOverall Aggregated After epoch {}\tAcc: {:.4f}\tMacro-F: {:.3f}\tMicro-F: {:.3f}\tWted: {:.3f}\n".format(
      *metrics
      ))

    if log_path:
        with open(log_path,'a') as f:
            f.write(
              "{:03d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(
                *metrics
              )
            )
    return metrics


def evalQuerySet(model, query, support, support_label, N_way, one_hot=False):
    model.eval()
    with torch.no_grad():
        q_feat, _, _, _ = model(query)
        cos0 = nn.CosineSimilarity(dim=1, eps=1e-6)

        all_classes = torch.arange(N_way)
       
        def findcosinesim(q):
            q = q.reshape(1,-1)
            random_supp = torch.tensor([]).to(device)
            for c in all_classes:
                s = torch.where(support_label==c)[0]
                sid = np.random.randint(0,s.size()[0])
                random_supp = torch.cat((random_supp, support[s[sid]].reshape(1,-1)), dim=0)

            q = q.to(device)
            random_supp = random_supp.to(device)
            cosine_sim = cos0(q, random_supp)
            

            delta = 1e-2
            maxi = torch.argmax(cosine_sim)
            max_cos = cosine_sim[maxi]
            pred_c = all_classes[maxi]
            collision = torch.abs(cosine_sim - max_cos) < delta

            if (torch.sum(collision) > 1):
                tries = 3
                while tries:
                    tries-=1
                    for c in all_classes[collision]:
                        s = torch.where(support_label==c)[0]
                        sid = np.random.randint(0,s.size()[0])
                        random_supp[c] = support[s[sid]].reshape(1,-1)
                    
                    random_supp = random_supp.to(device)
                    
                    cosine_sim = cos0(q, random_supp)
                    maxi = torch.argmax(cosine_sim)
                    max_cos = cosine_sim[maxi]
                    pred_c = all_classes[maxi]
                    collision = torch.abs(cosine_sim - max_cos) < delta
                    if (torch.sum(collision) == 1.0):
                      break
                
            if not one_hot:
                return cosine_sim

            y = torch.zeros(N_way)
            y[pred_c] = 1
            return y
            

        y_pred = map(findcosinesim, q_feat)
        y_pred = torch.stack(list(y_pred))
    return y_pred
    


if __name__ == "__main__":

    # to run
    # python train.py --epochs 10 --mode train --N 5 --K 5 --class-mapping path/image_class_labels.txt --image-mapping path/classes.txt --image-dir path_to_img_dir
    parser = argparse.ArgumentParser(description='PyTorch Implementation FSL-Image Classification')

    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--mode", type=str, default='test', help="with mode to use")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument(
        "--model_save_dir", type=str, default='./drive/MyDrive/CV_Project/Code/models/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='FSL_N5_K3',
        help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')
    parser.add_argument(
        '--N', type=int, default=5, help='N-way class')
    parser.add_argument(
        '--K', type=int, default=7, help='K-shot instances per class')
    parser.add_argument(
        '--class-mapping', type=str, default="./drive/MyDrive/CV_Project/Images/CUB_200_2011/CUB_200_2011/image_class_labels.txt", help="path to class to img mapping")
    parser.add_argument(
        '--classes-path', type=str, default="./drive/MyDrive/CV_Project/Images/CUB_200_2011/CUB_200_2011/classes.txt", help="path to img to filename mapping")
    parser.add_argument(
        '--image-dir', type=str, default="./drive/MyDrive/CV_Project/Images/CUB_200_2011/CUB_200_2011/images/", help="dir where all images are stored")
    parser.add_argument(
        '--log-path', type=str, default="./drive/MyDrive/CV_Project/Code/logs/", help="path to record metrics and training losses")
    

    args = parser.parse_args()


    print("Model Name: ",args.checkpoint,"N: ", args.N,"K: ", args.K, "Mode: ", args.mode)
                    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    

    model = FSL(N=args.N).to(device)

    if args.mode == 'train':
        train_dataset = EpiDataset(N=args.N, K=args.K, txt_file=args.class_mapping, classes_path=args.classes_path, image_dir=args.image_dir, data_set='train')
        val_dataset = EpiDataset(N=args.N, K=args.K, txt_file=args.class_mapping, classes_path=args.classes_path, image_dir=args.image_dir, data_set='val')
        test_dataset = EpiDataset(N=args.N, K=args.K, txt_file=args.class_mapping, classes_path=args.classes_path, image_dir=args.image_dir, data_set='test')

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory = True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        print("Data read")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        train_log_file = args.log_path + args.checkpoint+'_train_loss.txt'
        metric_log_file = args.log_path + args.checkpoint+'_metric_val.txt'

        with open(train_log_file, 'w') as f:
            f.write("Epoch\tAcc\tMSE_loss\tProto_loss\tCL_loss\n")
        with open(metric_log_file, 'w') as f:
            f.write("Epoch\tAcc\tMacro-F1\tMicro-F1\tWeighted-F1\n")

        print("Training begins")
        
        best_score = 0.0
        for epoch in range(1, args.epochs + 1):
            train(model, train_loader, optimizer, epoch=epoch, N_way=args.N, k_shot=args.K, log_interval=15,log_path = train_log_file)
            scheduler.step()
            print("\nValidation Set:\n")
            metrics = evalModel(model, val_loader, epoch, N_way=args.N, k_shot=args.K, log_path = metric_log_file)
            if metrics[2] > best_score:
                best_score = metrics[2]
                torch.save(model.state_dict(), args.model_save_dir + args.checkpoint + ".pt")
        
    elif args.mode == 'test':
        test_dataset = EpiDataset(N=args.N, K=args.K, txt_file=args.class_mapping, classes_path=args.classes_path, image_dir=args.image_dir, data_set='test')
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        print("Data read")

        model.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        evalModel(model, test_loader, 1, N_way=args.N, k_shot=args.K)

            
        
        
