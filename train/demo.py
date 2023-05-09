# import os
# import sys
# sys.path.append('../')
#
# from model.base_model import Encoder,Decoder
# import torch
# import json
# from torch.utils.data import DataLoader
# from torch.nn.utils.rnn import pack_padded_sequence
# from datasets import Dataset
# from torch.optim.lr_scheduler import StepLR
# import torchvision.transforms as transforms
# import torch.nn as nn
# import time
# from utils import *
# from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
# from torch.optim import lr_scheduler
# from evaluation import Cider as Cider1
# import numpy as np
# from beam_search import *
#
# embed_dim=512
# decoder_dim=512
# encoder_lr=5e-5
# decoder_lr=5e-5
# grad_clip = 5.
# d_model=512
# d_k=512
# d_v=512
# head=8
# reduction=32
# encoder_dim=1024
# attention_dim=512
#
#
#
# cider_tu=[]
# encoder=Encoder()
# decoder=Decoder(d_model=d_model,d_k=d_k,d_v=d_v,head=head,reduction=reduction,embed_dim=embed_dim,decoder_dim=decoder_dim,vocab_size=vocab_size,attention_dim=attention_dim,encoder_dim=encoder_dim,dropout=0.5)
#
# encoder_params=list(encoder.resnet152.parameters())+list(encoder.resnet50.parameters())
# encoder_optimizer=torch.optim.Adam(encoder_params,encoder_lr)
# decoder_optimizer=torch.optim.Adam(decoder.parameters(),decoder_lr)
#
#
# checkpoint_xe = torch.load('checkpoint_xe/checkpoint_best.pth')
# encoder.load_state_dict(checkpoint_xe['encoder'])
# decoder.load_state_dict(checkpoint_xe['decoder'])
# start_epoch = checkpoint_xe['start_epoch']
# encoder_optimizer.load_state_dict(checkpoint_xe['encoder_optimizer'])
# decoder_optimizer.load_state_dict(checkpoint_xe['decoder_optimizer'])
#
# print(encoder)
# print(decoder)
import torch
import torch.nn.functional as F

# import torch
#
# x = torch.randn(1,5, requires_grad=True)
# print(x)
# # x = F.softmax(x, dim=1)
# x = torch.nn.Softmax(x)
# print(x)
# l = 0
# for i in range(5):
#     l = l + x[0][i]
#
# print(l)
# l.backward()
# print(x.grad)
# result=torch.randn((3))
# zero=torch.zeros((1))
# while(result.shape[0]<5):
#     result=torch.cat([result,zero],dim=0)
# print(result)
x=torch.randn((4,5))
print(x)
print(x[:2,:])
