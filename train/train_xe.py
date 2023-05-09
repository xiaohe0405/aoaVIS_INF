import os

import sys
sys.path.append('../')
from model.base_model import Encoder,Decoder
import torch
import json
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
from datasets import Dataset
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
import torch.nn as nn
import time
from utilss import *

from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from torch.optim import lr_scheduler

import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

datajsonPath='data_4000_split.json'
wordmapPath='WORDMAP_4000_split.json'

writer=SummaryWriter("log_leakrelu")

directory='2000'
train_dataset=Dataset(split='TRAIN')
train_dataset.image_pre(directory)
train_dataset.caption_pre(datajsonPath=datajsonPath,wordmapPath=wordmapPath,max_len=50)
val_dataset=Dataset(split='TEST')
val_dataset.image_pre(directory)
val_dataset.caption_pre(datajsonPath=datajsonPath,wordmapPath=wordmapPath,max_len=50)

with open(wordmapPath, 'r') as j:
    word_map = json.load(j)
    vocab_size=len(word_map)

batch_size = 32
train_loader = DataLoader(dataset=train_dataset,
                         batch_size=batch_size,
                         shuffle=True)
val_loader = DataLoader(dataset=val_dataset,
                         batch_size=batch_size,
                         shuffle=True)

epochs=120
start_epoch=-1

embed_dim=512
decoder_dim=512
encoder_lr=1e-4
decoder_lr=1e-4
grad_clip = 5.
d_model=512
d_k=512
d_v=512
head=8
reduction=32
encoder_dim=1024
attention_dim=512


encoder=Encoder()
decoder=Decoder(d_model=d_model,d_k=d_k,d_v=d_v,head=head,reduction=reduction,embed_dim=embed_dim,decoder_dim=decoder_dim,vocab_size=vocab_size,attention_dim=attention_dim,encoder_dim=encoder_dim,dropout=0.5)
encoder=encoder.to(device)
decoder=decoder.to(device)

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
encoder_params=list(encoder.resnet152.parameters())+list(encoder.resnet50.parameters())

# encoder_optimizer=torch.optim.Adam(encoder_params,encoder_lr)
# decoder_optimizer=torch.optim.Adam(decoder.parameters(),decoder_lr)

encoder_optimizer=torch.optim.AdamW(encoder_params,lr=encoder_lr,betas=(0.9, 0.999), eps=1e-8)
decoder_optimizer=torch.optim.AdamW(decoder.parameters(),lr=decoder_lr,betas=(0.9, 0.999), eps=1e-8)

scheduler1 = StepLR(encoder_optimizer, step_size=10, gamma=0.5)
scheduler2 = StepLR(decoder_optimizer, step_size=10, gamma=0.5)

def train(train_loader,encoder,decoder,criterion,encoder_optimizer,decoder_optimizer,epoch):
    '''
    :param train_loader:
    :param encoder:
    :param decoder:
    :param criterion:
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param epoch:
    :return:
    '''
    encoder.train()
    decoder.train()
    batch_time=AverageMeter()
    data_time=AverageMeter()
    losses=AverageMeter()
    top5accs=AverageMeter()
    start=time.time()
    aaa = (epoch - 0) * len(train_loader)
    for i, (key,image_v, image_i, caption, valid_len) in enumerate(train_loader):
        image_v = image_v.to(device)
        image_i = image_i.to(device)
        caption = torch.tensor([item.cpu().detach().numpy() for item in caption]).to(device)
        caption = caption.to(device)
        valid_len = valid_len.to(device)

        visual,ir = encoder(image_v, image_i)
        # print(visual.shape)
        # print(ir.shape)
        scores = decoder(visual,ir, caption, valid_len)
        targets = caption[:, 1:]

        scores = pack_padded_sequence(scores, valid_len.cpu(), batch_first=True, enforce_sorted=False)[0].to(device)
        targets = pack_padded_sequence(targets, valid_len.cpu(), batch_first=True, enforce_sorted=False)[0].to(device)

        loss = criterion(scores, targets)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss.backward()

        # for x in encoder_optimizer.param_groups[0]['params']:
        #     print(x.grad)




        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)


        encoder_optimizer.step()
        decoder_optimizer.step()
        top5=accuracy(scores,targets,5)
        losses.update(loss.item(),sum(valid_len))
        top5accs.update(top5,sum(valid_len))
        batch_time.update(time.time()-start)
        start=time.time()

        train_loss.append(losses.val)
        train_acc.append(top5accs.val)
        writer.add_scalar("train_loss", losses.val, aaa + 1)
        writer.add_scalar("train_acc", top5accs.val, aaa + 1)
        aaa += 1
        x1.append(epoch)

        if i%10 ==0:
            print('Epoch:[{0}][{1}/{2}]\t'
                  f'Batch Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                  f'Data Load Time{data_time.val:.3f}({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f}({losses.avg:.4f})\t'
                  f'Top-5 Accuracy {top5accs.val:.3f}({top5accs.avg:.3f})'.format(epoch,i,len(train_loader),batch_time=batch_time,
                                                                          data_time=data_time,losses=losses,
                                                                  top5accs=top5accs))


    print(str(losses.val) + '   ' + str(top5accs.val))




def validate(val_loader,encoder,decoder,criterion):
    '''
    :param val_loader:
    :param encoder:
    :param decoder:
    :param criterion:
    :return:
    '''
    decoder.eval()
    encoder.eval()
    batch_time=AverageMeter()
    losses=AverageMeter()
    top5accs=AverageMeter()
    start=time.time()
    references=list()
    predictions=list()

    aaa = (epoch - 0) * len(val_loader)
    with torch.no_grad():
        for i, (key,image_v, image_i, caption, valid_len) in enumerate(val_loader):
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
            scores = decoder(visual,ir, caption, valid_len)
            targets = caption[:, 1:]
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, valid_len.cpu(), batch_first=True, enforce_sorted=False)[0].to(device)
            targets = pack_padded_sequence(targets, valid_len.cpu(), batch_first=True, enforce_sorted=False)[0].to(device)

            loss = criterion(scores, targets)

            losses.update(loss.item(),sum(valid_len))
            top5=accuracy(scores,targets,5)
            top5accs.update(top5,sum(valid_len))
            batch_time.update(time.time()-start)
            val_loss.append(losses.val)
            val_acc.append(top5accs.val)

            writer.add_scalar("val_loss", losses.val, aaa+1)
            writer.add_scalar("val_acc", top5accs.val, aaa+1)
            aaa+=1
            x2.append(epoch)
            start=time.time()


            if i % 10 == 0:
                print('Epoch:[{0}][{1}/{2}]\t'
                      f'Batch Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f}({losses.avg:.4f})\t'
                      f'Top-5 Accuracy {top5accs.val:.3f}({top5accs.avg:.3f})'.format(epoch, i, len(val_loader),
                                                                                      batch_time=batch_time,
                                                                                      losses=losses,
                                                                                      top5accs=top5accs))


            # print(scores_copy.shape)
            _, preds = torch.max(scores_copy, dim=2)
            # print('-----------------------------------------preds-------------------------------------------------------------')
            # print(preds.shape)
            preds = preds.tolist()
            # print(preds)
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:valid_len[j]])  # remove pads
                # print('-----------------------------------------temp_preds-------------------------------------------------------------')
                # print(temp_preds)
            preds = temp_preds
            predictions.extend(preds)

        bleu_sum=0.0
        for i in range(len(predictions)):
            bleu4 = sentence_bleu(references[i], predictions[i])
            bleu_sum+=bleu4
        bleu4_mean=bleu_sum/len(predictions)

        print('\n * LOSS - {losses.avg:.3f}, TOP-5 ACCURACY - {top5accs.avg:.3f}, BLEU-4 - {bleu4}\n'.format(
            losses=losses,
            top5accs=top5accs,
            bleu4=bleu4_mean))

    print(str(losses.val) + '   ' + str(top5accs.val))
    return bleu4_mean


# def evaluate(val_loader,encoder,decoder):
#     references = list()
#     predictions = list()
#     with torch.no_grad():
#         for i, (key,image_v, image_i, caption, valid_len) in enumerate(val_loader):
#             print(i)
#             keys=list(key)
#             caption_list=[]
#             for k in keys:
#                 l=val_dataset.find_imageId(k,datajsonPath)
#                 caption_list.append(l)
#             references.extend(caption_list)
#
#             image_v = image_v.to(device)
#             image_i = image_i.to(device)
#             caption = torch.tensor([item.cpu().detach().numpy() for item in caption]).to(device)
#             caption = caption.to(device)
#             valid_len = valid_len.to(device)
#             visual,ir = encoder(image_v, image_i)
#
#             #每一个visual 和ir 采样一个 句子回来
#             seq_res=beam_seq_val(visual, ir, decoder, beam_size=5)
#             # print(seq_res)
#
#             seqs = {}
#             for i in range(len(caption_list)):
#                 # ref 为['','','','']
#                 ref = caption_list[i]
#                 l = []
#                 for r in ref:
#                     seq = [w for w in r if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
#                     words = [rev_word_map[ind] for ind in seq]
#                     words = ' '.join(list(map(str, words)))
#                     l.append(words)
#                 seqs.update({str(i): l})
#
#             # print(seqs)
#             scorer=Scorer(seq_res,seqs)
#             total_scores=scorer.compute_scores()
#             c=total_scores['CIDEr']
#             print(c)
#             CIDEr_sum.append(c)
#         print('-------------------------平均------------------------------------')
#         avg_cider=sum(CIDEr_sum) / len(CIDEr_sum)
#         print(avg_cider)
#         return avg_cider
#     # references = list()
#     # hypotheses = list()
#     # print(len(val_loader))
#     # for i, (key, image_v, image_i, caption, valid_len) in enumerate(val_loader):
#     #     keys = list(key)
#     #     caption_list = []
#     #     for k in keys:
#     #         l = val_dataset.find_imageId(k, datajsonPath)
#     #         caption_list.append(l)
#     #     references.extend(caption_list)
#     #
#     #     k = 5
#     #     image_v = image_v.to(device)
#     #     image_i = image_i.to(device)
#     #     visual, ir = encoder(image_v, image_i)
#     #
#     #     batch_size = visual.size(0)
#     #     enc_image_size = visual.size(-1)
#     #     visual = visual.transpose(1, 3)
#     #     ir = ir.transpose(1, 3)
#     #
#     #     visual = decoder.fc_1(visual)
#     #     ir = decoder.fc_2(ir)
#     #     visual = decoder.relu(visual)
#     #     ir = decoder.relu(ir)
#     #
#     #     visual = visual.transpose(1, 3)
#     #     ir = ir.transpose(1, 3)
#     #
#     #     enc_out = decoder.self_channel_atten(visual, ir).transpose(1, 3)  # (batch_size,14,14,1024)
#     #     enc_dim = enc_out.size(-1)
#     #     enc_out = enc_out.contiguous().view(batch_size, -1, enc_dim)
#     #
#     #     num_pixels = enc_out.size(1)
#     #     enc_out = enc_out.expand(k, num_pixels, enc_dim)
#     #
#     #     k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)
#     #
#     #     seqs = k_prev_words
#     #     top_k_scores = torch.zeros(k, 1).to(device)
#     #
#     #     complete_seqs = list()
#     #     complete_seqs_scores = list()
#     #
#     #     # 开始解码
#     #     step = 1
#     #     h, c = decoder.init_hidden_state(enc_out)
#     #
#     #
#     #     while True:
#     #         embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s,emded_dim)
#     #         Aoa_encoding, alpha = decoder.aoa_atten(enc_out, embeddings)
#     #
#     #         h, c = decoder.decode_step(torch.cat([embeddings, Aoa_encoding], dim=1), (h, c))
#     #         scores = decoder.fc(h)
#     #         scores = F.log_softmax(scores, dim=1)
#     #         scores = top_k_scores.expand_as(scores) + scores  # (s,vocab_size)
#     #         if step == 1:
#     #             top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
#     #         else:
#     #             top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
#     #
#     #         prev_word_inds = top_k_words / vocab_size  # (s)
#     #         next_word_inds = top_k_words % vocab_size  # (s)
#     #
#     #         seqs = torch.cat([seqs[prev_word_inds.long()], next_word_inds.unsqueeze(1)], dim=1)  # (s,step+1)
#     #
#     #         incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
#     #                            next_word != word_map['<end>']]
#     #         complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
#     #
#     #         if len(complete_inds) > 0:
#     #             complete_seqs.extend(seqs[complete_inds].tolist())
#     #             # complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
#     #             complete_seqs_scores.extend(top_k_scores[complete_inds])
#     #         k -= len(complete_inds)  # reduce beam length accordingly
#     #         if k == 0:
#     #             break
#     #         seqs = seqs[incomplete_inds]
#     #         h = h[prev_word_inds[incomplete_inds].long()]
#     #         c = c[prev_word_inds[incomplete_inds].long()]
#     #         enc_out = enc_out[prev_word_inds[incomplete_inds].long()]
#     #         top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
#     #         k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
#     #
#     #         # Break if things have been going on too long
#     #         if step > 50:
#     #             break
#     #         step += 1
#     #
#     #     # i = complete_seqs_scores.index(max(complete_seqs_scores))
#     #     # seq = complete_seqs[i]
#     #
#     #     if complete_seqs_scores:
#     #         i = complete_seqs_scores.index(max(complete_seqs_scores))
#     #         seq = complete_seqs[i]
#     #
#     #     else:
#     #         seq = [word_map['<unk>']]
#     #
#     #     hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
#     #
#     # refer = {}
#     # for i in range(len(references)):
#     #     ll = []
#     #     for key in references[i]:
#     #         l = ''
#     #         for k in range(len(key)):
#     #             if k == 0:
#     #                 l = l + rev_word_map[key[k]]
#     #             else:
#     #                 l = l + ' ' + rev_word_map[key[k]]
#     #         ll.append(l)
#     #     refer[i] = ll
#     #
#     # hypoth = {}
#     # # print(hypotheses)
#     # for i in range(len(hypotheses)):
#     #     a = []
#     #     l = ''
#     #     for k in range(len(hypotheses[i])):
#     #         if k == 0:
#     #             l = l + rev_word_map[hypotheses[i][k]]
#     #         else:
#     #             l = l + ' ' + rev_word_map[hypotheses[i][k]]
#     #     a.append(l)
#     #     hypoth[i] = a
#     #
#     # scores, _ = evaluation.compute_scores(refer, hypoth)
#     # print(scores)
#     # return scores['CIDEr']
#     #
#     # scorer = Scorer(hypoth, refer)
#     # total_scores = scorer.compute_scores()
#     # return total_scores['CIDEr']



if __name__=='__main__':
    file_counts = 0
    best_bleu=0.0
    file_lists=os.listdir('checkpoint_xe')
    file_lists.sort(key=lambda fn: os.path.getmtime('checkpoint_xe/' + fn) if not os.path.isdir('checkpoint_xe/' + fn) else 0)

    if os.path.isdir("checkpoint_xe"):
        for dirpath, dirnames, filenames in os.walk('checkpoint_xe'):
            file_counts = len(filenames)
    if file_counts!=0:
        checkpoint=torch.load('checkpoint_xe/checkpoint_last.pth')
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        start_epoch = checkpoint['start_epoch']
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        scheduler1.load_state_dict(checkpoint['scheduler1']),
        scheduler2.load_state_dict(checkpoint['scheduler2']),
        train_loss=checkpoint['train_loss']
        best_bleu = checkpoint['best_bleu']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint['val_loss']
        train_acc = checkpoint['train_acc']
        val_acc = checkpoint['val_acc']
        bleu = checkpoint['bleu']
        x1 = checkpoint['x1']
        x2 = checkpoint['x2']
        x = checkpoint['x']

    for epoch in range(start_epoch + 1, epochs):
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        scheduler1.step()
        scheduler2.step()

        bleu4=validate(val_loader=val_loader,
                 encoder=encoder,
                 decoder=decoder,
                 criterion=criterion)

        writer.add_scalar("bleu4", bleu4, epoch)


        start_epoch=epoch


        checkpoint = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            'scheduler1': scheduler1.state_dict(),
            'scheduler2': scheduler2.state_dict(),
            "start_epoch": start_epoch,
            'train_loss': train_loss,
            'best_bleu':best_bleu,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'bleu': bleu,
            'x1': x1,
            'x2': x2,
            'x': x
        }
        print(bleu4)
        x.append(epoch)
        bleu.append(bleu4)
        print(encoder_optimizer.state_dict()['param_groups'][0]['lr'])
        print(decoder_optimizer.state_dict()['param_groups'][0]['lr'])
        if bleu4>best_bleu:
            best_bleu=bleu4
            checkpoint = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'scheduler1': scheduler1.state_dict(),
                'scheduler2': scheduler2.state_dict(),
                "start_epoch": start_epoch,
                'train_loss': train_loss,
                'best_bleu': best_bleu,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'bleu': bleu,
                'x1': x1,
                'x2': x2,
                'x': x
            }
            torch.save(checkpoint, 'checkpoint_xe/checkpoint_best.pth')
        torch.save(checkpoint, 'checkpoint_xe/checkpoint_last.pth')
    writer.close()


