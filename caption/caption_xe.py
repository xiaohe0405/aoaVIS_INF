import torch
from model.base_model import Encoder,Decoder
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms

from scipy.misc import imread,imresize


import warnings
warnings.filterwarnings('ignore')

# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
datajsonPath='../train/data_4000_split.json'
wordmapPath='../train/WORDMAP_4000_split.json'
with open(wordmapPath,'r') as j:
    word_map=json.load(j)
    vocab_size=len(word_map)
rev_word_map={v:k for k,v in word_map.items()}

epochs=100
embed_dim=512
decoder_dim=512
encoder_lr=5e-5
decoder_lr=5e-5
grad_clip = 5
d_model=512
d_k=512
d_v=512
head=8
reduction=32
encoder_dim=1024
attention_dim=512


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder=Encoder().to(device)
decoder=Decoder(d_model=d_model,d_k=d_k,d_v=d_v,head=head,reduction=reduction,embed_dim=embed_dim,decoder_dim=decoder_dim,vocab_size=vocab_size,attention_dim=attention_dim,encoder_dim=encoder_dim,dropout=0.5).to(device)
encoder.eval()
decoder.eval()

def caption_image_beam_search(encoder,decoder,img_visual_path,img_ir_path,word_map,beam_size):
    '''
    :param encoder:编码器
    :param decoder:解码器
    :param img_visual_path:自然图像路径
    :param img_ir_path:红外图像路径
    :param word_map:词典
    :param beam_size:束搜索大小
    :return:描述序列
    '''
    k=beam_size
    vocab_size=len(word_map)

    image_v=imread(img_visual_path)
    image_i=imread(img_ir_path)
    if len(image_v.shape)==2:
        img=image_v[:,:,np.newaxis]
        image_v=np.concatenate([img,img,img],axis=2)
    if len(image_i.shape)==2:
        img=image_i[:,:,np.newaxis]
        image_i=np.concatenate([img,img,img],axis=2)
    image_v=imresize(image_v,(224,224))
    image_v=image_v.transpose(2,0,1)
    image_i = imresize(image_i, (224, 224))
    image_i = image_i.transpose(2, 0, 1)

    image_v=image_v/255.
    image_i = image_i / 255.
    image_v=torch.FloatTensor(image_v).to(device)
    image_i = torch.FloatTensor(image_i).to(device)
    transform_v = transforms.Compose([
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])
    transform_i = transforms.Compose([
                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])])
    image_v = transform_v(image_v)
    image_i = transform_i(image_i)
    image_v=image_v.unsqueeze(0)
    image_i = image_i.unsqueeze(0)


    visual, ir = encoder(image_v, image_i)  # (batch_size,2048,14,14)  torch.Size([5, 512, 14, 14])
    # encoder_dim=enc_out.size(3)
    # decoder(visual,ir,caption)
    batch_size = visual.size(0)
    enc_image_size = visual.size(-1)
    visual = visual.transpose(1, 3)   #(batch_size,14,14,2048)  torch.Size([5, 14, 14, 2048])
    ir = ir.transpose(1, 3)

    # (batch_size,enc_image_size,enc_image_size,2048)
    visual = decoder.fc_1(visual)
    ir = decoder.fc_2(ir)
    visual = decoder.relu(visual)
    ir = decoder.relu(ir)
    # (batch_size,enc_image_size,enc_image_size,512)
    visual = visual.transpose(1, 3)
    ir = ir.transpose(1, 3)

    enc_out = decoder.self_channel_atten(visual, ir).transpose(1, 3)  # (batch_size,14,14,1024)
    enc_dim = enc_out.size(-1)
    enc_out = enc_out.contiguous().view(batch_size, -1, enc_dim)
    # num_pixels=enc_out.size(1)
    # enc_out=enc_out.view(1,-1,encoder_dim)
    num_pixels = enc_out.size(1)
    enc_out = enc_out.expand(k, num_pixels, enc_dim)

    # k_prev_words=torch.Tensor([[word_map['<start>']]]*k).to(device)
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
    seqs=k_prev_words
    top_k_scores=torch.zeros(k,1).to(device)

    complete_seqs=list()
    complete_seqs_scores=list()

    #开始解码
    step=1
    h,c=decoder.init_hidden_state(enc_out)

    # enc_out = enc_out.mean(dim=1)
    # encoder_out = decoder.enc_fc(encoder_out)

    while True:
        embeddings=decoder.embedding(k_prev_words).squeeze(1)    #(s,emded_dim)
        Aoa_encoding, alpha = decoder.aoa_atten(enc_out, embeddings)
        # attention_weighted_encoding,alpha,beta_t=self.attention(embeddings[:batch_size_t,t,:],enc_out[:batch_size_t,:],
        #                                                         h[:batch_size_t,:],c[:batch_size_t,:])
        # Aoa_encoding = Aoa_encoding.sum(1)
        # attention_weighted_encoding, alpha, beta_t = decoder.attention(embeddings,enc_out,h,c)
        h,c=decoder.decode_step(torch.cat([embeddings,Aoa_encoding],dim=1),(h,c))
        scores=decoder.fc(h)
        scores=F.softmax(scores,dim=1)
        scores=top_k_scores.expand_as(scores)+scores   #(s,vocab_size)
        if step==1:
            top_k_scores,top_k_words=scores[0].topk(k,0,True,True)  #(s)
        else:
            top_k_scores,top_k_words=scores.view(-1).topk(k,0,True,True)   #(s)


        prev_word_inds=top_k_words/vocab_size   #(s)
        next_word_inds=top_k_words%vocab_size    #(s)

        seqs=torch.cat([seqs[prev_word_inds.long()],next_word_inds.unsqueeze(1)],dim=1)  #(s,step+1)



        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            # complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds].long()]
        c = c[prev_word_inds[incomplete_inds].long()]
        enc_out = enc_out[prev_word_inds[incomplete_inds].long()]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]

    return seq


if __name__=='__main__':
    checkpoint = torch.load('../train/checkpoint_xe/checkpoint_best.pth',map_location=device)
    # checkpoint = torch.load('../train/checkpoint_scst_greed/checkpoint_last.pth', map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    img_visual_path=r'14857630_1.bmp'
    img_ir_path=r'14857630_2.bmp'
    beam_size=5

    encoder=encoder.to(device)
    encoder.eval()
    decoder=decoder.to(device)
    decoder.eval()

    seq=caption_image_beam_search(encoder,decoder,img_visual_path,img_ir_path,word_map,beam_size)
    words = [rev_word_map[ind] for ind in seq]
    words=' '.join(words)
    print(words)