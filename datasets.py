import string
import pandas as pd
import torch
import json
import os
from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
import h5py

def random_pick(some_list, probabilities):
    '''
    根据概率划分数据集
    :param some_list:
    :param probabilities:
    :return:
    '''
    x = random.uniform(0,1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item

def preprocess_nmt(text):
    '''
    进行文本预处理
    1、用空格 代替不间断空格
    2、小写字母代替大写字母
    3、在单词和标点符号之间插入空格
    '''
    def no_space(char,prev_char):
        return char in set(',.!?') and prev_char !=' '
    # 用空格 代替不间断空格  小写转大写
    text = text.translate(str.maketrans('', '', string.punctuation))
    text=text.replace('\u202f',' ').replace('\xa0',' ').lower()
    # 在单词和标点符号之间 加入空格
    out=[' '+char if i >0 and no_space(char,text[i-1]) else char for i,char in enumerate(text)]
    return ''.join(out)

def tokenize(text):
    '''
    分词：将文本分成list
    '''
    list=text.split()
    return list

def createJson(filepath):
    '''
    :param filepath: xlsx的路径
    :return: json文件（图片和描述一一对用）
    '''
    data=pd.read_excel(filepath)
    data=pd.DataFrame(data)
    #设置imgid以及sentids
    # imgid=0
    sentid=0
    sentids = []
    sentences =[]
    probabilities=[0.6,0.2,0.2]
    some_list=['TRAIN','VAL','TEST']
    jsontext = {'images': [],'dataset':"visible_ir"}
    for index, row in data.iterrows():
        filename=row['filename']
        imgid=filename.split('.')[0]
        split=random_pick(some_list,probabilities)
        sentids.append(sentid)
        sentence = {'tokens': tokenize(preprocess_nmt(row['caption'])), 'raw': row['caption'], 'imgid': imgid, 'sentid': sentid}
        sentences.append(sentence)
        if (index+1)%5==0:
            jsontext['images'].append({'imgid':imgid,'split':split,'filename':row['filename'],'sentids':sentids,'sentences':sentences})
            # imgid+=1
            sentids=[]
            sentences =[]
        sentid+=1

    res=json.dumps(jsontext, indent=4, separators=(',', ': '))
    return res

def createWordMap(filepath,output_folder,max_len,min_word_freq=5):
    '''
    根据词频创建字典
    :param filepath: json输入文件路径
    :param outputpath: json输出文件路径
    :return:
    '''
    word_freq=Counter()
    with open(filepath, 'r') as j:
        data = json.load(j)
    for img in data['images']:
        captions=[]
        for c in img['sentences']:
            word_freq.update(c['tokens'])
            if len(c['tokens'])<=max_len:
                captions.append(c['tokens'])
    #创建字典
    words=[w for w in word_freq.keys() if word_freq[w]>min_word_freq]
    word_map={k:v+1 for v,k in enumerate(words)}
    word_map['<unk>']=len(word_map)+1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>']=0

    # Save word map to a JSON
    with open(os.path.join(output_folder, 'WORDMAP_4000_split.json'), 'w') as j:
        json.dump(word_map, j)

class Dataset(Dataset):
    def __init__(self,split):
        super(Dataset, self).__init__()
        self.split=split
        self.images={}
        self.caption_list_all=[]
        self.word_map=[]
        self.max_len=50
        self.datajsonPath=''
        self.wordmapPath=''
    def image_pre(self,directory):
        path_v=directory+'/Visual'
        path_i=directory+'/IR'
        #自然图像处理
        loader_v = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        for name in os.listdir(path_v):
            image=Image.open(path_v+'/'+name)
            image=loader_v(image)
            image_id = name.split('.')[0]
            self.images.setdefault(image_id,[]).append(image)
        #红外图像处理
        loader_i = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        for name in os.listdir(path_i):
            image=Image.open(path_i+'/'+name)
            image=loader_i(image)
            if image.size(0)==1:
                image=image.repeat(3,1,1)
            load=transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
            image=load(image)
            image_id = name.split('.')[0]
            self.images.setdefault(image_id, []).append(image)

    def caption_pre(self,datajsonPath,wordmapPath,max_len):
        '''
        :param datajsonPath:
        :param wordmapPath:
        :return:
        '''
        self.max_len=max_len
        self.datajsonPath=datajsonPath
        self.wordmapPath=wordmapPath
        with open(datajsonPath, 'r') as j:
            data = json.load(j)
        with open(wordmapPath, 'r') as j:
            self.word_map = json.load(j)

        for key in data['images']:
            if key['split']!=self.split:
                continue
            for index in key['sentences']:
                captions = {}
                c=index['tokens']
                enc_c = [self.word_map['<start>']] + [self.word_map.get(word, self.word_map['<unk>']) for word in c]\
                        + [self.word_map['<end>']]\
                        + [self.word_map['<pad>']] * (max_len - len(c))
                captions[index['imgid']]=enc_c
                self.caption_list_all.append(captions)

    #根据imageid  寻找参考的5的句子
    def find_imageId(self,imageId,datajsonPath):
        '''
        :param imageId: 图像Id 一个图像对应5个caption
        :param datajsonPath: 存储数据的json路径
        :return: 返回该imageId 对应的[[],[],[],[],[]]
        '''
        datajsonPath=self.datajsonPath
        with open(datajsonPath,'r') as j:
            data = json.load(j)
        l=[]
        for key in data['images']:
            if key['imgid']==imageId:
                for index in key['sentences']:
                    ll=index['tokens']
                    lll = [self.word_map.get(k, self.word_map['<unk>']) for k in ll]
                    l.append(lll)
        return l

    def __getitem__(self, index):
        # self.caption_list_all=self.caption_list_all[:10]
        dict=self.caption_list_all[index]
        # caption_list = []
        for key,value in dict.items():
            # list_5=self.find_imageId(key,self.datajsonPath)
            # l_5=[]
            # for i in range(len(list_5)):
            #     l=[self.word_map.get(k, self.word_map['<unk>']) for k in list_5[i]]
            #     l_5.append(l)
            image_v=self.images.get(key)[0]
            # print(image_v.shape)
            image_i = self.images.get(key)[1]
            # print(image_i.shape)
            sum=0
            for i in value:
                sum+=1
                if i==self.word_map['<end>']:
                    valid_len=sum
                    sum=0
            caption=value
        # caption_list=torch.LongTensor(caption_list)
        if self.split=='TRAIN':
            return key,image_v,image_i,torch.LongTensor(caption),valid_len
        else:
            return key,image_v,image_i,torch.LongTensor(caption),valid_len

    def __len__(self):
        return len(self.caption_list_all)

    def collate_fn(data):
        '''
        :param:data list of tuple(image_v,image_i,caption,valid_len)
                    image_v:Tensor(3,224,224)
                    image_i:Tensor(3,224,224)
                    caption:list
                    valid_len:int
        :return:
        '''
        # Sort a data list by caption length (descending order).
        # data.sort(key=lambda x: len(x[1]), reverse=True)
        image_v, image_i,caption,valid_len = zip(*data)

        # Merge images (from tuple of 3D tensor to 4D tensor).
        image_v = torch.stack(image_v, 0)
        image_i = torch.stack(image_i, 0)
        # print(image_v.shape)
        # print(image_i.shape)
        lengths = [len(cap) for cap in caption]
        valid_len=list(valid_len)
        # print(len(valid_len))
        # targets = torch.zeros(len(caption), max(lengths)).long()
        list_all=[]
        for _, cap in enumerate(caption):
            # end = lengths[i]
            list_all.append(cap)
            # targets[i, :] = torch.Tensor(cap)
        caption=torch.tensor(list_all)
        # print(caption.shape)
        # caption=torch.tensor(np.array(caption,dtype=np.int32))
        # caption = torch.stack(caption, 0)
        return image_v,image_i,caption,valid_len

#重新 从 HDF5文件里 读取图像 数据 减少一下IO的时间
class MydataSet(torch.utils.data.Dataset):
    def __init__(self,split,datajsonPath,wordmapPath,transform=None):
        super(MydataSet, self).__init__()
        self.split = split
        self.images = {}
        self.caption_list_all = []
        self.word_map = []
        self.max_len = 50
        self.datajsonPath = datajsonPath
        self.wordmapPath = wordmapPath
        self.trainval_v=h5py.File('../images_trainval_v.hdf5','r')
        self.trainval_i=h5py.File('../images_trainval_i.hdf5','r')
        self.test_v=h5py.File('../images_test_v.hdf5','r')
        self.test_i=h5py.File('../images_test_i.hdf5','r')
        self.transform=transform
    def caption_pre(self):
        '''
        :param datajsonPath:
        :param wordmapPath:
        :return:
        '''
        with open(self.datajsonPath, 'r') as j:
            data = json.load(j)
        with open(self.wordmapPath, 'r') as j:
            self.word_map = json.load(j)

        for key in data['images']:
            if key['split']!=self.split:
                continue
            for index in key['sentences']:
                captions = {}
                c=index['tokens']
                enc_c = [self.word_map['<start>']] + [self.word_map.get(word, self.word_map['<unk>']) for word in c]\
                        + [self.word_map['<end>']]\
                        + [self.word_map['<pad>']] * (self.max_len - len(c))
                captions[index['imgid']]=enc_c
                self.caption_list_all.append(captions)

    def find_imageId(self,imageId,datajsonPath):
        '''
        :param imageId: 图像Id 一个图像对应5个caption
        :param datajsonPath: 存储数据的json路径
        :return: 返回该imageId 对应的[[],[],[],[],[]]
        '''
        datajsonPath=self.datajsonPath
        with open(datajsonPath,'r') as j:
            data = json.load(j)
        l=[]
        for key in data['images']:
            if key['imgid']==imageId:
                for index in key['sentences']:
                    ll=index['tokens']
                    lll = [self.word_map.get(k, self.word_map['<unk>']) for k in ll]
                    l.append(lll)
        return l

    def __getitem__(self, index):
        dict=self.caption_list_all[index]
        for key, value in dict.items():
            if self.split=='TEST':
                image_v = self.test_v[key][:]
                image_i = self.test_i[key][:]
            else:
                image_v=self.trainval_v[key][:]
                image_i = self.trainval_i[key][:]

            sum=0
            for i in value:
                sum+=1
                if i==self.word_map['<end>']:
                    valid_len=sum
                    sum=0
            caption=value

        if self.split == 'TRAIN':
            return key,image_v, image_i, torch.LongTensor(caption), valid_len
        else:
            return key, image_v, image_i, torch.LongTensor(caption), valid_len

    def __len__(self):
        return len(self.caption_list_all)





# if __name__=='__main__':
#     transform=transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225])
#     ])
#     datajsonPath='data_4000_split.json'
#     wordmapPath='WORDMAP_4000_split.json'
#     train=MydataSet(split='VAL',datajsonPath=datajsonPath,wordmapPath=wordmapPath,transform=transform)
#     train.caption_pre()
#     train_loader = DataLoader(dataset=train,
#                              batch_size=20,
#                              shuffle=True)
#     print(len(train))
#     for i,(key,image_v,image_i,caption,valid_len) in enumerate(train_loader):
#         print(image_v.shape)
#         print(image_i.shape)









#
#
# filepath='2000.xlsx'
#
# f = open('data_4000_split.json', 'w')
# f.write(createJson(filepath))
# f.close()
#
# createWordMap('data_4000_split.json','',48,2)

#
# dataset=Dataset(split='val')
# directory='500'
# # datajsonPath='data_500_2500_split.json'
# # wordmapPath='WORDMAP_500_2500_split.json'
# # dataset.image_pre(directory)
# # dataset.caption_pre(datajsonPath,wordmapPath,max_len=48)
# #
# # print(dataset.__len__())
# directory='2000'
# # # #
# datajsonPath='data_4000_split.json'
# wordmapPath='WORDMAP_4000_split.json'
# # train_dataset=Dataset(split='train')
# val_dataset=Dataset(split='test')
# # # # #
# val_dataset.image_pre(directory)
# val_dataset.caption_pre(datajsonPath,wordmapPath,max_len=48)
# # #
# for key in val_dataset.images:
#     if val_dataset.images.get(key)[0].shape!=(3,224,224):
#         print(val_dataset.images.get(key)[0].shape)
#     if val_dataset.images.get(key)[1].shape!=(3,224,224):
#         print(val_dataset.images.get(key)[1].shape)
# #
# # l=train_dataset.caption_list_all
# # for i in l:
# #     for key, value in i.items():
# #         if len(value)!=50:
# #             print(key)
# #             print(value)
#
# # print(train_dataset.images)
# val_dataset.image_pre(directory)
# val_dataset.caption_pre(datajsonPath,wordmapPath,max_len=48)
# ll=train_dataset.caption_list_all
# for i in ll:
#     for key, value in i.items():
#         if len(value)!=50:
#             print(key)
#             print(value)
#
# l=train_dataset.caption_list_all
# for i in l:
#     print(i)
#
# print(len(train_dataset))
# train_loader = DataLoader(dataset=train_dataset,
#                          batch_size=20,
#                          shuffle=True)
# val_loader = DataLoader(dataset=val_dataset,
#                          batch_size=100,
#                          shuffle=True)
# for i,(image_v, image_i, caption, valid_len) in enumerate(train_loader):
#     print('------------------------------')
#     print(valid_len)
#     print(image_i.shape)
#     print('------------------------------')
# file='2000/IR'
#
#
# loader_i = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor()
#         ])
# for name in os.listdir(file):
#     image=Image.open(file+'/'+name)
#     image=loader_i(image)
#     if image.size(0)==1:
#         image=image.repeat(3,1,1)
#     load=transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
#     image=load(image)
#     print(image.shape==(3,224,224))

# data_loader = DataLoader(dataset=dataset,
#                          batch_size=64,
#                          shuffle=False)
# #
# #
# for i,(image_v, image_i, caption,valid_len,c) in enumerate(data_loader):
#     print(valid_len)



