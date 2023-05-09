import sys
sys.path.append('../')

from model.base_model import Encoder,Decoder
import torch
import torch.nn.functional as F
from datasets import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import json
import warnings
warnings.filterwarnings('ignore')




from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.spice.spice import Spice


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])

datajsonPath='../train/data_4000_split.json'
wordmapPath='../train/WORDMAP_4000_split.json'

directory='../train/2000'
test_dataset=Dataset(split='TEST')
test_dataset.image_pre(directory)
test_dataset.caption_pre(datajsonPath=datajsonPath,wordmapPath=wordmapPath,max_len=50)


with open(wordmapPath,'r') as j:
    word_map=json.load(j)
    vocab_size=len(word_map)
rev_word_map={v:k for k,v in word_map.items()}


test_loader=DataLoader(dataset=test_dataset,
                       batch_size=1,
                       shuffle=True)

epochs=100
start_epoch=-1


embed_dim=512
decoder_dim=512
encoder_lr=1e-5
decoder_lr=1e-5
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



def evaluate(beam_size):
    references = list()
    hypotheses = list()
    print(len(test_loader))
    for i,(key,image_v,image_i,caption,valid_len) in enumerate(test_loader):
        keys=list(key)
        caption_list = []
        for k in keys:
            l = test_dataset.find_imageId(k, datajsonPath)
            caption_list.append(l)
        references.extend(caption_list)

        k=beam_size
        image_v=image_v.to(device)
        image_i=image_i.to(device)
        visual, ir = encoder(image_v, image_i)
        # encoder_dim=enc_out.size(3)
        # decoder(visual,ir,caption)
        batch_size = visual.size(0)
        enc_image_size = visual.size(-1)
        visual = visual.transpose(1, 3)
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

        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)

        complete_seqs = list()
        complete_seqs_scores = list()

        # 开始解码
        step = 1
        h, c = decoder.init_hidden_state(enc_out)

        # enc_out = enc_out.mean(dim=1)
        # encoder_out = decoder.enc_fc(encoder_out)

        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s,emded_dim)
            Aoa_encoding, alpha = decoder.aoa_atten(enc_out, embeddings)
            # attention_weighted_encoding,alpha,beta_t=self.attention(embeddings[:batch_size_t,t,:],enc_out[:batch_size_t,:],
            #                                                         h[:batch_size_t,:],c[:batch_size_t,:])
            # Aoa_encoding = Aoa_encoding.sum(1)
            # attention_weighted_encoding, alpha, beta_t = decoder.attention(embeddings,enc_out,h,c)
            h, c = decoder.decode_step(torch.cat([embeddings, Aoa_encoding], dim=1), (h, c))
            scores = decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)
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
            enc_out = enc_out[prev_word_inds[incomplete_inds].long()]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        # i = complete_seqs_scores.index(max(complete_seqs_scores))
        # seq = complete_seqs[i]

        if complete_seqs_scores:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            seq = [word_map['<end>']]

        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

    refer = {}
    for i in range(len(references)):
        ll=[]
        for key in references[i]:
            l = ''
            for k in range(len(key)):
                if k == 0:
                    l = l + rev_word_map[key[k]]
                else:
                    l = l + ' ' + rev_word_map[key[k]]
            ll.append(l)
        refer[i] =ll

    hypoth={}
    # print(hypotheses)
    for i in range(len(hypotheses)):
        a = []
        l = ''
        for k in range(len(hypotheses[i])):
            if k==0:
                l=l+rev_word_map[hypotheses[i][k]]
            else:
                l = l + ' ' + rev_word_map[hypotheses[i][k]]
        a.append(l)
        hypoth[i]=a
    print(refer)
    print(hypoth)
    return refer,hypoth
    # scores, _ = evaluation.compute_scores(refer, hypoth)
    # print(scores)
    # return scores['CIDEr']


class Scorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
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
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.3f" % (method, score))
                total_scores[method] = score

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))

#
if __name__ == '__main__':
    checkpoint = torch.load('../train/checkpoint_xe/checkpoint_best.pth', map_location=device)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    encoder = encoder.to(device)
    encoder.eval()
    decoder = decoder.to(device)
    decoder.eval()
    beam_size = 5
    refer, hypoth=evaluate(beam_size)
    # scores, _ = evaluation.compute_scores(refer, hypoth)
    # print(scores)
    # return scores
    scorer = Scorer(hypoth, refer)
    total_scors=scorer.compute_scores()




