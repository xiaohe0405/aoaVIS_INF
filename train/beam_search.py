import random

import torch
import json
import torch
import torch.nn.functional as F
import utils
import numpy as np
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from torch.distributions import Categorical




datajsonPath='data_4000_split.json'
wordmapPath='WORDMAP_4000_split.json'
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#字典
with open(wordmapPath, 'r') as j:
    word_map = json.load(j)
    vocab_size=len(word_map)
rev_word_map={v:k for k,v in word_map.items()}


# def beam_search(visual,ir,references,decoder,beam_size=5):
#     seq_l=[]
#     k=beam_size
#     batch_size = visual.size(0)
#     enc_image_size = visual.size(-1)
#     visual = visual.transpose(1, 3)
#     ir = ir.transpose(1, 3)
#
#     # (batch_size,enc_image_size,enc_image_size,2048)
#     visual = decoder.fc_1(visual)
#     ir = decoder.fc_2(ir)
#     visual = decoder.relu(visual)
#     ir = decoder.relu(ir)
#     # (batch_size,enc_image_size,enc_image_size,512)
#     visual = visual.transpose(1, 3)
#     ir = ir.transpose(1, 3)
#
#
#     enc_out = decoder.self_channel_atten(visual, ir).transpose(1, 3)  # (batch_size,14,14,1024)
#     enc_dim = enc_out.size(-1)
#     enc_out = enc_out.contiguous().view(batch_size, -1, enc_dim)
#
#
#     # num_pixels=enc_out.size(1)
#     # enc_out=enc_out.view(1,-1,encoder_dim)
#     num_pixels = enc_out.size(1)
#
#     for batch_id in range(batch_size):
#         k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
#         enc=enc_out[batch_id]
#         enc = enc.unsqueeze(0).expand(k, num_pixels, enc_dim)
#         seq = beam(enc, k_prev_words, decoder, beam_size=5)
#         seq_l.append(seq)
#
#     return seq_l
#




'''
传入 可见光[batch_size, 2018, 14, 14]和红外图像特征[batch_size, 512, 14, 14]
返回 50个句子 以json形式返回  {'0':[],'1':[]}
'''
def beam_seq(vis,inf,decoder,beam_size):
    batch_size=vis.size(0)
    # k = beam_size
    enc_image_size = vis.size(-1)
    vis = vis.transpose(1, 3)
    inf = inf.transpose(1, 3)

    # (batch_size,enc_image_size,enc_image_size,2048)
    vis = decoder.fc_1(vis)
    inf = decoder.fc_2(inf)
    vis = decoder.relu(vis)
    inf = decoder.relu(inf)
    # (batch_size,enc_image_size,enc_image_size,512)
    vis = vis.transpose(1, 3)
    inf = inf.transpose(1, 3)

    enc_out = decoder.self_channel_atten(vis, inf).transpose(1, 3)  # (batch_size,14,14,1024)
    enc_dim = enc_out.size(-1)
    enc_out = enc_out.contiguous().view(batch_size, -1, enc_dim)

    # num_pixels=enc_out.size(1)
    # enc_out=enc_out.view(1,-1,encoder_dim)
    num_pixels = enc_out.size(1)
    seq_res={}

    scores=[]
    # scores=
    # scores=torch.zeros((batch_size,5))
    count = 0
    log_probs=[]
    outputs=[]
    for index in range(batch_size):
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * beam_size).to(device)  # (k, 1)
        enc = enc_out[index]
        enc = enc.unsqueeze(0).expand(beam_size, num_pixels, enc_dim)
        '''
        h, c = decoder.init_hidden_state(enc)
        t=0
        max_len=20
        seqs = k_prev_words
        log_prob = []
        output=[]
        while(t<max_len):
            embeddings = decoder.embedding(seqs).squeeze(1)  # (s,emded_dim)
            # attention_weighted_encoding, alpha, beta_t = decoder.attention(embeddings, enc, h, c)
            Aoa_encoding, alpha = decoder.aoa_atten(enc, embeddings)
            h, c = decoder.decode_step(torch.cat([embeddings, Aoa_encoding], dim=1), (h, c))
            scores = decoder.fc(h)
            # prob = F.softmax(scores, dim=1)
            m = Categorical(logits=scores)
            action = m.sample()
            output.append(action)
            # print(res)
            word=m.log_prob(action)
            log_prob.append(word)
            # seqs=res.long()
            t+=1
            seqs=action.long()
        log_probs.append(torch.stack(log_prob))
        outputs.append(torch.stack(output))
    outputs=torch.stack(outputs).view(batch_size,5,max_len).contiguous().cpu().numpy().tolist()
    log_probs=torch.stack(log_probs).view(batch_size,5,max_len).contiguous()

    count=0
    for i in range(batch_size):
        for seq in outputs[i]:
            seq = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            words = [rev_word_map[ind] for ind in seq]
            words = [' '.join(list(map(str, words)))]
            seq_res.update({count:words})
            count+=1
        '''

        score,seqs,_=beam(enc, k_prev_words, decoder, beam_size=beam_size)

        if beam_size==5:
            while len(seqs)<beam_size:
                lens=len(seqs)
                if lens==1:
                    i=0
                elif lens==0:
                    seqs.append([word_map['<unk>']])
                    i=0
                else:
                    i=random.randint(0,lens-1)
                seqs.append(seqs[i])

            while len(score) < beam_size:
                lens = len(score)
                if lens == 1:
                    i = 0
                elif lens == 0:
                    score.append(torch.tensor(-0.0,requires_grad=True).to(device))
                    i = 0
                else:
                    i = random.randint(0, lens - 1)
                score.append(score[i])
        for seq in seqs:
            #删掉起始符号 和结束符号
            seq=[w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
            words = [rev_word_map[ind] for ind in seq]
            words = [' '.join(list(map(str, words)))]
            seq_res.update({count:words})
            count+=1
        scores.append(torch.stack(score))
        # scores.append(score)
    # print(scores)
    return scores,seq_res


def beam_seq_val(vis,inf,decoder,beam_size=5):
    '''
    :param vis:
    :param inf:
    :param decoder:
    :param beam_size:
    :return:
    '''
    batch_size = vis.size(0)
    k = beam_size
    enc_image_size = vis.size(-1)
    vis = vis.transpose(1, 3)
    inf = inf.transpose(1, 3)

    # (batch_size,enc_image_size,enc_image_size,2048)
    vis = decoder.fc_1(vis)
    inf = decoder.fc_2(inf)
    vis = decoder.relu(vis)
    inf = decoder.relu(inf)
    # (batch_size,enc_image_size,enc_image_size,512)
    vis = vis.transpose(1, 3)
    inf = inf.transpose(1, 3)

    enc_out = decoder.self_channel_atten(vis, inf).transpose(1, 3)  # (batch_size,14,14,1024)
    enc_dim = enc_out.size(-1)
    enc_out = enc_out.contiguous().view(batch_size, -1, enc_dim)

    # num_pixels=enc_out.size(1)
    # enc_out=enc_out.view(1,-1,encoder_dim)
    num_pixels = enc_out.size(1)
    seq_res = {}
    count = 0
    scores = []
    for index in range(batch_size):
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
        enc = enc_out[index]
        enc = enc.unsqueeze(0).expand(k, num_pixels, enc_dim)
        _, _, seq = beam(enc, k_prev_words, decoder, beam_size=k)
        seq = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        words = [rev_word_map.get(ind) for ind in seq]
        words = [' '.join(list(map(str, words)))]

        seq_res.update({str(index):words})
    # print(scores.shape)
    return seq_res



def beam(enc,k_prev_words,decoder,beam_size):
    k=beam_size
    seqs = k_prev_words
    top_k_scores = torch.zeros(k, 1).to(device)
    results=[]
    # a=torch.zeros(5).to(device)
    complete_seqs = list()
    complete_seqs_scores = list()
    # 开始解码
    step = 1
    h, c = decoder.init_hidden_state(enc)

    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s,emded_dim)
        # attention_weighted_encoding, alpha, beta_t = decoder.attention(embeddings, enc, h, c)
        Aoa_encoding, alpha = decoder.aoa_atten(enc, embeddings)
        # attention_weighted_encoding,alpha,beta_t=self.attention(embeddings[:batch_size_t,t,:],enc_out[:batch_size_t,:],
        #                                                         h[:batch_size_t,:],c[:batch_size_t,:])
        # Aoa_encoding = Aoa_encoding.sum(1)
        # attention_weighted_encoding, alpha, beta_t = decoder.attention(embeddings,enc_out,h,c)


        h, c = decoder.decode_step(torch.cat([embeddings, Aoa_encoding], dim=1), (h, c))

        # h, c = decoder.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c))
        scores = decoder.fc(h)
        scores = F.softmax(scores, dim=1)


        # print('----------------softmax 后的scores-----------------')
        #
        # result,_=scores.max(dim=1)
        # result = torch.log(result)
        # # #
        # zero = torch.zeros((1)).to(device)
        # while (result.shape[0] < 5):
        #     result = torch.cat([result, zero], dim=0)
        # # result=result.expand_as(a)

        # print(result)
        # results.append(result)
        # print(results)

        scores = top_k_scores.expand_as(scores) + scores  # (s,vocab_size)


        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        prev_word_inds = top_k_words / vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s,step+1)

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
        enc = enc[prev_word_inds[incomplete_inds].long()]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    l = []
    # # print('==============================================')
    # # print(scores.shape)
    if complete_seqs_scores:
        j = complete_seqs_scores.index(max(complete_seqs_scores))
        seq=complete_seqs[j]
        for i in complete_seqs_scores:
            l.append(i)
    else:
        seq=[word_map['<end>']]
        # for i in range(beam_size):
        #     # top_k_scores.append(seq)
        #     l.append(0.0)
    # print('-----------------------complete_seqs_scores 以及scores的值------------------------')
    # print(complete_seqs_scores)
    # print(complete_seqs_scores)
    # seq=complete_seqs_scores.index(max(complete_seqs_scores))
    # print('-------------------------------top_k_scores------------------------------------')
    # print(top_k_scores.shape)
    # print(top_k_scores)
    #
    # print('-------------------------------scores------------------------------------')
    # print(scores.mean(dim=1))
    # print(scores.shape)
    #
    # print('-------------------------------complete_seqs_scores------------------------------------')
    # print(complete_seqs_scores.shape)
    # print(complete_seqs_scores)
    # print(results)

    # results=torch.stack(results)
    # # print(results.shape)
    # # print(results.mean(dim=0))
    #
    # if(results.shape[0]<20):
    #     z=torch.zeros((20-results.shape[0],5)).to(device)
    #     results = torch.cat([results, z], dim=0)
    # if(results.shape[0]>20):
    #     results=results[:20,:]

    # print('-----------------------------results----------------------------------------')
    # print(results.shape)
    # print(results)
    return l,complete_seqs,seq
