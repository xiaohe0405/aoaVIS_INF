import os
import sys

from model.base_model import Encoder,Decoder
import torch
import json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from datasets import Dataset
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
import torch.nn as nn
import time
from utils import *
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from torch.optim import lr_scheduler
from evaluation import Cider as Cider1
import numpy as np
from beam_search import *

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider as Cider
import os


import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datajsonPath='data_4000_split.json'
wordmapPath='WORDMAP_4000_split.json'

directory='2000'
train_dataset=Dataset(split='TRAIN')
train_dataset.image_pre(directory)
train_dataset.caption_pre(datajsonPath=datajsonPath,wordmapPath=wordmapPath,max_len=50)
val_dataset=Dataset(split='VAL')
val_dataset.image_pre(directory)
val_dataset.caption_pre(datajsonPath=datajsonPath,wordmapPath=wordmapPath,max_len=50)

with open(wordmapPath, 'r') as j:
    word_map = json.load(j)
    vocab_size=len(word_map)
rev_word_map={v:k for k,v in word_map.items()}

batch_size = 32
train_loader = DataLoader(dataset=train_dataset,
                         batch_size=batch_size,
                         shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                         batch_size=batch_size,
                         shuffle=True)

epochs=100
start_epoch=-1

embed_dim=512
decoder_dim=512
encoder_lr=5e-5
decoder_lr=5e-5
grad_clip = 5.
d_model=512
d_k=512
d_v=512
head=8
reduction=32
encoder_dim=1024
attention_dim=512



cider_tu=[]
encoder=Encoder()
decoder=Decoder(d_model=d_model,d_k=d_k,d_v=d_v,head=head,reduction=reduction,embed_dim=embed_dim,decoder_dim=decoder_dim,vocab_size=vocab_size,attention_dim=attention_dim,encoder_dim=encoder_dim,dropout=0.5)


# x=range(0,epochs)
x=[]
x1=[]
x2=[]

train_loss=[]
val_loss=[]
bleu=[]
train_acc=[]
val_acc=[]


resume=False

encoder=encoder.to(device)
decoder=decoder.to(device)

criterion=nn.CrossEntropyLoss().to(device)
encoder_params=list(encoder.resnet50.parameters())+list(encoder.resnet18.parameters())

encoder_optimizer=torch.optim.Adam(encoder_params,encoder_lr)
decoder_optimizer=torch.optim.Adam(decoder.parameters(),decoder_lr)

scheduler1 = StepLR(encoder_optimizer, step_size=10, gamma=0.8)
scheduler2 = StepLR(decoder_optimizer, step_size=10, gamma=0.8)

class Scorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        # print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            # (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),
            # (WMD(), "WMD"),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            # print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                # for sc, scs, m in zip(score, scores, method):
                    # print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                # print("%s: %0.3f" % (method, score))
                total_scores[method] = score

        # print('*****DONE*****')
        # for key, value in total_scores.items():
        #     print('{}:{}'.format(key, value))
        return total_scores


def train(train_loader,encoder,decoder,encoder_optimizer,decoder_optimizer,epoch):
    running_reward = .0
    running_reward_baseline = .0
    running_loss = .0

    for i, (key,image_v, image_i, caption, valid_len) in enumerate(train_loader):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        print(i)
        references = list()
        hypotheses = list()
        keys = list(key)
        caption_list = []
        for k in keys:
            l = train_dataset.find_imageId(k, datajsonPath)
            caption_list.append(l)
        references.extend(caption_list)
        image_v = image_v.to(device)
        image_i = image_i.to(device)
        caption = torch.tensor([item.cpu().detach().numpy() for item in caption]).to(device)
        caption = caption.to(device)
        valid_len = valid_len.to(device)
        batch_size=image_v.size(0)

        visual, ir = encoder(image_v, image_i)
        '''
        传入可见光图像[batch_size, 2018, 14, 14]和 红外图像[batch_size, 512, 14, 14]
        返回 batch*5个句子
        '''
        scores,seq_res=beam_seq(visual,ir,decoder,beam_size=5)
        scores=scores.requires_grad_().to(device)
        #根据真实标签  以字典的形式 返回
        seqs={}
        for i in range(len(references)*5):
            #ref 为['','','','']
            ref=references[i // 5]
            l=[]
            for r in ref:
                seq = [w for w in r if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
                words = [rev_word_map[ind] for ind in seq]
                words=' '.join(list(map(str, words)))
                l.append(words)
            seqs.update({i:l})

        reward = cider.compute_score(seqs, seq_res)[1].astype(np.float32)
        reward = torch.from_numpy(reward).to(device).view(batch_size, 5).requires_grad_()
        reward_baseline = torch.mean(reward, -1, keepdim=True).requires_grad_()
        # torch.mean(scores, -1)
        loss = (scores) * (reward - reward_baseline)
        loss.requires_grad_()
        # loss = -(torch.mean(scores)) * (reward - reward_baseline)
        loss = loss.mean()

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        running_loss += loss.item()
        running_reward += reward.mean().item()
        running_reward_baseline += reward_baseline.mean().item()
        print('-------------------loss------------reward-----------------reward_baseline-------------------')
        print(loss.item()/batch_size)
        print(reward.mean().item())
        print(reward_baseline.mean().item())
        # if i%10==0:
    loss = running_loss / len(train_loader)
    reward = running_reward / len(train_loader)
    reward_baseline = running_reward_baseline / len(train_loader)
    print(loss,reward,reward_baseline)
    return loss,reward,reward_baseline


def validate(val_loader,encoder,decoder):
    references = list()
    predictions = list()
    CIDEr_sum=[]
    with torch.no_grad():
        for i, (key,image_v, image_i, caption, valid_len) in enumerate(val_loader):
            print(i)
            keys=list(key)
            caption_list=[]
            for k in keys:
                l=val_dataset.find_imageId(k,datajsonPath)
                caption_list.append(l)
            references.extend(caption_list)

            image_v = image_v.to(device)
            image_i = image_i.to(device)
            caption = torch.tensor([item.cpu().detach().numpy() for item in caption]).to(device)
            caption = caption.to(device)
            valid_len = valid_len.to(device)
            visual,ir = encoder(image_v, image_i)

            #每一个visual 和ir 采样一个 句子回来
            seq_res=beam_seq_val(visual, ir, decoder, beam_size=5)
            # print(seq_res)

            seqs = {}
            for i in range(len(caption_list)):
                # ref 为['','','','']
                ref = caption_list[i]
                l = []
                for r in ref:
                    seq = [w for w in r if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
                    words = [rev_word_map[ind] for ind in seq]
                    words = ' '.join(list(map(str, words)))
                    l.append(words)
                seqs.update({str(i): l})

            # print(seqs)
            scorer=Scorer(seq_res,seqs)
            total_scores=scorer.compute_scores()
            c=total_scores['CIDEr']
            print(c)
            CIDEr_sum.append(c)
        print('-------------------------平均------------------------------------')
        avg_cider=sum(CIDEr_sum) / len(CIDEr_sum)
        print(avg_cider)
        return avg_cider



            # scorer.compute_scores()
            # print(scorer)

if __name__=='__main__':
    # 加载XE的 断点
    train_loss=[]
    x=[]
    checkpoint_xe = torch.load('checkpoint_aoa/checkpoint_last.pth')
    encoder.load_state_dict(checkpoint_xe['encoder'])
    decoder.load_state_dict(checkpoint_xe['decoder'])
    start_epoch = checkpoint_xe['start_epoch']
    encoder_optimizer.load_state_dict(checkpoint_xe['encoder_optimizer'])
    decoder_optimizer.load_state_dict(checkpoint_xe['decoder_optimizer'])


    #加载权重
    file_counts = 0
    best_cider = 0.0
    file_lists = os.listdir('checkpoint_scst')
    file_lists.sort(
        key=lambda fn: os.path.getmtime('checkpoint_scst/' + fn) if not os.path.isdir('checkpoint_scst/' + fn) else 0)

    if os.path.isdir("checkpoint_scst"):
        for dirpath, dirnames, filenames in os.walk('checkpoint_scst'):
            file_counts = len(filenames)
    if file_counts != 0:
        checkpoint = torch.load('checkpoint_scst/checkpoint_last.pth')
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        start_epoch = checkpoint['start_epoch']
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        cider_tu = checkpoint['cider_tu']
        best_cider= checkpoint['best_cider'],
        train_loss=checkpoint['train_loss']
        x=checkpoint['x']


    for epoch in range(start_epoch + 1, epochs):
        loss,reward,reward_baseline=train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)
        train_loss.append(loss)
        x.append(epoch)

        avg_cider=validate(val_loader=val_loader,
                 encoder=encoder,
                 decoder=decoder)

        start_epoch=epoch

        checkpoint = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            "start_epoch": start_epoch,
            'best_cider': best_cider,
            'cider_tu': cider_tu,
            'x':x,
            'train_loss':train_loss
        }
        print(best_cider)

        cider_tu.append(best_cider)
        print(encoder_optimizer.state_dict()['param_groups'][0]['lr'])
        print(decoder_optimizer.state_dict()['param_groups'][0]['lr'])
        if avg_cider > best_cider:
            best_cider = avg_cider
            checkpoint = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                # 'scheduler1': scheduler1.state_dict(),
                # 'scheduler2': scheduler2.state_dict(),
                "start_epoch": start_epoch,
                'best_cider': best_cider,
                'cider_tu': cider_tu,
                'x': x,
                'train_loss': train_loss
            }
            torch.save(checkpoint, 'checkpoint_scst/checkpoint_best.pth')
        torch.save(checkpoint, 'checkpoint_scst/checkpoint_last.pth')