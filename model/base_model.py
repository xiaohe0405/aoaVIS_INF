import torch
from torch import nn
import torchvision
from model.AttentionCompose import Aoa,self_Channel_Atten
device=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

from torch.nn import init
torchvision.models

class Encoder(nn.Module):
    '''
    Encoder
    '''
    def __init__(self,encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size=encoded_image_size
        #使用resnet101  pretrained=True返回在ImageNet上预训练过的模型
        '''
        resnet152=torchvision.models.resnet152(pretrained=True)
        #移除resnet的线性层和池化层 并且自适应池化 输出固定大小的向量
        modules152=list(resnet152.children())[:-2]
        self.resnet152=nn.Sequential(*modules152)

        #使用resnet50  用于提取红外图像特征
        resnet50=torchvision.models.resnet50(pretrained=False)
        modules50=list(resnet50.children())[:-2]
        self.resnet50=nn.Sequential(*modules50)

        self.adaptive_pool = nn.AdaptiveMaxPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()
        '''
        resnet152=torchvision.models.resnet152(pretrained=True)
        #移除resnet的线性层和池化层 并且自适应池化 输出固定大小的向量
        modules152=list(resnet152.children())[:-2]
        self.resnet152=nn.Sequential(*modules152)

        #使用resnet50  用于提取红外图像特征
        resnet50=torchvision.models.resnet50(pretrained=False)
        modules50=list(resnet50.children())[:-2]
        self.resnet50=nn.Sequential(*modules50)

        self.adaptive_pool = nn.AdaptiveMaxPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()
    def forward(self,imagesVisual,imagesIR):
        '''
        :param imagesVisual: (batch_size,3,image_size,image_size)
        :param imagesIR: (batch_size,3,image_size,image_size)
        :return:
        '''
        visual=self.resnet152(imagesVisual)  #(batch_size,2048,image_size/32,image_size/32)
        IR=self.resnet50(imagesIR)

        visual=self.adaptive_pool(visual)    #(batch_size,2048,encoded_image_size,encoded_image_size)
        IR=self.adaptive_pool(IR)
        return visual,IR
        # (batch_size,2048,14,14)  torch.Size([5, 512, 14, 14])
    def fine_tune(self,fine_tune=True):
        #微调resnet2-4个block 因为第一个块 提取的低层特征 不需要计算梯度了
        for p in self.resnet152.parameters():
            p.requires_grad=False
        for c in list(self.resnet152.children())[5:]:
            for p in c.parameters():
                p.requires_grad=fine_tune
        for p in self.resnet50.parameters():
            p.requires_grad=True



class Decoder(nn.Module):
    def __init__(self,d_model,d_k,d_v,head,reduction,embed_dim,decoder_dim,vocab_size,attention_dim,encoder_dim=1024,dropout=0.5):
        '''
        :param d_model: 自注意力机制中的d_model
        :param d_k:
        :param d_v:
        :param head: 多头自注意力机制 的头的个数
        :param reduction: 通道注意力机制  中间两个全连接层的 channels//reduction
        :param embed_dim: 词嵌入的大小
        :param decoder_dim: 解码器 hidden state cell的维度
        :param vocab_size: 词典的大小
        :param encoder_dim: 编码器 出来的可见光/红外图像的通道数
        :param dropout:
        '''
        super(Decoder, self).__init__()
        # self.attention_compose=AttentionCompose(block_num,channel,reduction)
        self.d_model=d_model
        self.d_k=d_k
        self.d_v=d_v
        self.head=head
        self.reduction=reduction
        self.embed_dim=embed_dim
        self.decoder_dim=decoder_dim
        self.vocab_size=vocab_size
        self.dropout=dropout
        self.encoder_dim = encoder_dim
        self.embedding=nn.Embedding(vocab_size,embed_dim)
        self.dropout=nn.Dropout(p=self.dropout)

        # 组合注意力机制
        self.self_channel_atten = self_Channel_Atten(d_model, d_k, d_v, head, reduction, dropout)
        #(batch_size,1024,14,14)
        self.aoa_atten=Aoa(encoder_dim=encoder_dim,decoder_dim=decoder_dim,attention_dim=attention_dim,dropout=dropout)

        self.init_h=nn.Linear(d_model*2,decoder_dim)
        self.init_c=nn.Linear(d_model*2,decoder_dim)
        self.fc_1=nn.Linear(2048,d_model)
        self.fc_2=nn.Linear(2048,d_model)

        self.relu=nn.ReLU()
        self.decode_step=nn.LSTMCell(embed_dim+attention_dim,decoder_dim,bias=True)
        self.fc=nn.Linear(decoder_dim,vocab_size)
        self.init_weight()


    def init_weight(self):
        self.embedding.weight.data.uniform_(-0.1,0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc_1.weight.data.uniform_(-0.1, 0.1)
        self.fc_1.bias.data.fill_(0)
        self.fc_2.weight.data.uniform_(-0.1, 0.1)
        self.fc_2.bias.data.fill_(0)


    def load_pretrained_embeddings(self,embeddings):
        self.embedding.weight=nn.Parameter(embeddings)

    def fine_tune_embeddings(self,fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad=fine_tune

    def init_hidden_state(self,encoder_out):
        mean_encoder_out=encoder_out.mean(dim=1)
        h=self.init_h(mean_encoder_out)
        c=self.init_c(mean_encoder_out)
        return h,c

    def forward(self,visual,ir,encoded_captions,valid_len):
        '''
        :param visual: (batch_size,1024,14,14)
        :param ir: (batch_size,512,14,14)
        :param encoded_captions:
        :param valid_len:
        :return:
        '''
        batch_size=visual.size(0)
        enc_image_size=visual.size(-1)

        self.enc_image_size=enc_image_size

        visual=visual.transpose(1,3)  #(batch_size,14,14,1024)
        ir=ir.transpose(1,3)  #(batch_size,14,14,512)

        #(batch_size,enc_image_size,enc_image_size,2048)
        visual=self.fc_1(visual)  #(batch_size,14,14,512)
        ir=self.fc_2(ir)          #(batch_size,14,14,512)
        visual=self.relu(visual)
        ir=self.relu(ir)

        # (batch_size,enc_image_size,enc_image_size,512)
        visual = visual.transpose(1, 3) #(batch_size,512,14,14)
        ir = ir.transpose(1, 3)         #(batch_size,512,14,14)

        #通道 自注意力融合后的结果
        enc_out=self.self_channel_atten(visual,ir).transpose(1,3)  #(batch_size,14,14,1024)


        enc_dim=enc_out.size(-1)

        enc_out=enc_out.contiguous().view(batch_size,-1,enc_dim)   #(batch_size,196,1024)
        num_pixels=enc_out.size(1)

        # enc_out=self.enc_fc(enc_out)  #(batch_size,196,decoder_dim)

        vocab_size=self.vocab_size
        embeddings=self.embedding(encoded_captions)   #(batch_size,max_length,embed_dim)

        #初始化lstm的隐藏状态 和记忆细胞的状态
        h,c=self.init_hidden_state(enc_out)   #(batch_size,decoder_dim)
        predictions=torch.zeros(batch_size, max(valid_len), vocab_size)
        #
        # enc_out=enc_out.mean(dim=1)
        for t in range(max(valid_len)):
            batch_size_t = sum([l > t for l in valid_len])
            Aoa_encoding, alpha=self.aoa_atten(enc_out[:batch_size_t,:],embeddings[:batch_size_t,t,:])
            # attention_weighted_encoding,alpha,beta_t=self.attention(embeddings[:batch_size_t,t,:],enc_out[:batch_size_t,:],
            #                                                         h[:batch_size_t,:],c[:batch_size_t,:])
            # Aoa_encoding=Aoa_encoding.sum(1)
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], Aoa_encoding[:batch_size_t,:]], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
        return predictions



# if __name__=='__main__':
#     vis=torch.randn((5,3,224,224))
#     inf=torch.randn((5,3,224,224))
#     encoder=Encoder(encoded_image_size=14)
#     vis,inf=encoder(vis,inf)
#
#     decoder=Decoder(d_model=512,d_k=512,d_v=512,head=8,reduction=16,embed_dim=512,decoder_dim=512,vocab_size=1000,attention_dim=1024,encoder_dim=1024,dropout=0.5)
#
#     print(vis.shape)
#     print(inf.shape)




