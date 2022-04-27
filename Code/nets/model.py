import torch.nn as nn
import torchvision.models as tmodels
from nets.autoencoder import AutoEncoder
from nets.self_attention import SelfAttention
import torch.nn.functional as F


'''
1. Get resnet features
2. Pass through AE -> x -> reconstruct
3. self attention (x) -> contrastive
4. classifier -> Randomly pick 1 from each class, cosine sim
              -> pick max (in case of ties, pick sample from tied class again)
'''
class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.view(x.size(0), -1)

class resNet_extractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        m = tmodels.resnet18(pretrained=False)
        m = nn.Sequential(*(list(m.children())[:-1]))
        # for param in m.parameters():
        #   param.requires_grad = False
        self.resnet = m
        # self.enc = nn.Sequential(
        #     self.conv_block(3, 64),
        #     self.conv_block(64, 64),
        #     self.conv_block(64, 32),
        #     self.conv_block(32, 32),
        #     # self.conv_block(16, 16),
        #     Flatten()
        #   )
    def conv_block(self, in_channels, out_channels):
      return nn.Sequential(
          nn.Conv2d(in_channels, out_channels, 3, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(),
          nn.MaxPool2d(2)
          )

    def forward(self, x):
        return self.resnet(x)


class FSL(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.resnet = resNet_extractor()
        self.autoencoder = AutoEncoder(input_shape = 512)
        self.self_attn = SelfAttention(d_model=128,nhead=1,dim_feedforward=128)
        # self.fc = nn.Linear(6272,512)
        # self.fc1 = nn.Linear(128, 64)
        # self.fc2 = nn.Linear(128,kwargs["N"])

    def forward(self, img_tensor):
        # 512-d img feature
        img_feat = self.resnet(img_tensor).squeeze()
        # print(img_feat.shape)
        # img_feat = self.fc(img_feat)
        # img_feat = F.normalize(img_feat, p=2, dim=1)
        bottleneck_feat, reconstructed_feat = self.autoencoder(img_feat)

        self_attn_feat = self.self_attn(bottleneck_feat)
        # x = self.fc1(self_attn_feat)
        # x = F.relu(x)
        # x = self.fc2(self_attn_feat)
        
        return self_attn_feat, img_feat, reconstructed_feat, self_attn_feat




