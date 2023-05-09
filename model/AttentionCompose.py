import numpy as np
import torch
from torch import nn
from torch.nn import init

class SEAttention(nn.Module):
    '''
    通道注意力机制
    '''
    def __init__(self,channel=512,reduction=16):
        super(SEAttention, self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.fc=nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Linear(channel//reduction,channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                init.kaiming_normal_(m.weight,mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                init.constant_(m.weight,1)
                init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                init.normal_(m.weight,std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias,0)

    def forward(self,x):
        b,c,_,_=x.size()   #b 为batch_size, c为channel
        y1=self.avg_pool(x).view(b,c)
        y1=self.fc(y1).view(b,c,1,1)
        y2=self.max_pool(x).view(b,c)
        y2=self.fc(y2).view(b,c,1,1)
        y=y1+y2
        y=self.sigmoid(y)
        return x*y.expand_as(x)

class ScaledDotProductAttention(nn.Module):
    '''
    带缩放的自注意力
    '''
    def __init__(self,d_model,d_k,d_v,head,dropout=0.5):
        '''
        :param d_model: encoded_dim
        :param d_k: Q和K 的维数
        :param d_v: V的维数
        :param head: 多头注意力的头数
        :param dropout:
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q=nn.Linear(d_model,head*d_k)
        self.fc_k=nn.Linear(d_model,head*d_k)
        self.fc_v=nn.Linear(d_model,head*d_v)
        self.fc_o=nn.Linear(head*d_v,d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model=d_model
        self.d_k=d_k
        self.d_v=d_v
        self.head=head
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                init.kaiming_normal_(m.weight,mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                init.constant_(m.weight,1)
                init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                init.normal_(m.weight,std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias,0)

    def forward(self,queries,keys,values,attention_mask=None,attention_weights=None):
        '''
        :param queries: 查询向量 是红外图像的特征 (batch_size,nq,d_model)  num_piexls  nk nv nk 都应该是196
        :param keys: 键向量 是自然图像的特征 (batch_size,nk,d_model)
        :param values: 值向量 是自然图像的特征 (batch_size,nv,d_model)
        :param attention_mask:
        :param attention_weights:权重参数 (batch_size,head,nq,nk)
        :return:
        '''
        batch_size,nq=queries.shape[:2]
        nk=keys.shape[1]

        Q=self.fc_q(queries).view(batch_size,nq,self.head,self.d_k).permute(0,2,1,3) #(batch_size,head,nq,d_k)

        K=self.fc_k(keys).view(batch_size,nk,self.head,self.d_k).permute(0,2,3,1)  #(batch_size,head,d_k,nk)
        V=self.fc_v(values).view(batch_size,nk,self.head,self.d_v).permute(0,2,1,3) #(batch_size,head,nk,d_v)


        att=torch.matmul(Q,K)/np.sqrt(self.d_k)  #(batch_size,head,nq,nk)
        if attention_weights is not None:
            att=att*attention_weights
        if attention_mask is not None:
            att=att.masked_fill(attention_mask,-np.inf)
        att=torch.softmax(att,-1)
        # print('-------------------att----------------')
        # print(att.shape)
        # print('-------------------V----------------')
        # print(V.shape)
        att=self.dropout(att)
        #(batch_size,nq,head,d_v)
        out=torch.matmul(att,V).permute(0,2,1,3).contiguous().view(batch_size,nq,self.head*self.d_v)
        # print('-------------------out-------------------------')
        # print(out.shape)
        out=self.fc_o(out)  #(batch_size,nq,d_model)
        return out

class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # nn.GELU(),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class self_Channel_Atten(nn.Module):
    def __init__(self,d_model,d_k,d_v,head,reduction=16,drop=0.5):
        super(self_Channel_Atten, self).__init__()
        self.self_v_i=ScaledDotProductAttention(d_model=d_model,d_k=d_k,d_v=d_v,head=head,dropout=drop)
        self.self_i_v=ScaledDotProductAttention(d_model=d_model,d_k=d_v,d_v=d_k,head=head,dropout=drop)
        self.se=SEAttention(channel=d_model*2,reduction=reduction)
        self.layernorm=nn.LayerNorm(d_model)
        self.ffn=FFN(d_model,d_model//reduction,dropout=drop)
    def forward(self,visual,ir):
        '''
        :param visual: #(batch_size,512,14,14)
        :param ir: #(batch_size,512,14,14)
        :return:
        '''
        visual=visual.transpose(1,3)  #(batch_size,14,14,512)
        ir=ir.transpose(1,3)  #(batch_size,14,14,512)
        batch_size = visual.size(0)
        d_model = visual.size(-1)
        enc_image_size=visual.size(1)

        visual=visual.contiguous().view((batch_size,-1,d_model))#(batch_size,196,512)
        ir = ir.contiguous().view((batch_size, -1, d_model))#(batch_size,196,512)

        self_atten_v=self.self_v_i(ir,visual,visual)
        self_atten_i=self.self_i_v(visual,ir,ir)

        #残差 和层归一化
        self_atten_v =self.layernorm(self_atten_v+visual)
        self_atten_v=self.layernorm(self.ffn(self_atten_v)+self_atten_v)#(batch_size,196,512)

        self_atten_i = self.layernorm(self_atten_i+ir)
        self_atten_i = self.layernorm(self.ffn(self_atten_i) + self_atten_i)#(batch_size,196,512)

        self_atten_v=self_atten_v.view((batch_size,enc_image_size,enc_image_size,d_model)).transpose(1,3).contiguous()
        self_atten_i = self_atten_i.view((batch_size, enc_image_size, enc_image_size,d_model)).transpose(1,3).contiguous()
        x=torch.cat((self_atten_v,self_atten_i),dim=1)
        y=self.se(x)
        return y   #torch.Size([3, 1024, 14, 14])



class SotfAtten(nn.Module):
    def __init__(self,encoder_dim,decoder_dim,attention_dim,dropout=0.5):
        super(SotfAtten, self).__init__()
        self.fc_encoder=nn.Linear(encoder_dim,attention_dim)
        self.fc_decoder=nn.Linear(decoder_dim,attention_dim)
        self.full_atten=nn.Linear(attention_dim,1)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(p=dropout)
        self.softmax=nn.Softmax(dim=1)
    def forward(self,enc_out,word):
        enc_out=self.fc_encoder(enc_out)
        word=self.fc_decoder(word)
        y=self.full_atten(self.relu(enc_out+word.unsqueeze(dim=1))).squeeze(2)
        alpha=self.softmax(y)
        attention_weighted_encoding=(alpha.unsqueeze(dim=2)*enc_out).sum(1)
        return attention_weighted_encoding,alpha   #[natch_size,512]

class Aoa(nn.Module):
    def __init__(self,encoder_dim,decoder_dim,attention_dim,dropout=0.5):
        super(Aoa, self).__init__()
        self.atten=SotfAtten(encoder_dim=encoder_dim,decoder_dim=decoder_dim,attention_dim=attention_dim,dropout=dropout)
        self.fc_sig=nn.Linear(attention_dim*2,attention_dim)
        self.fc = nn.Linear(attention_dim * 2, attention_dim)
        self.sigmoid=nn.Sigmoid()
        self.dropout=nn.Dropout()
        self.relu=nn.ReLU()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                init.kaiming_normal_(m.weight,mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias,0)
            elif isinstance(m,nn.BatchNorm2d):
                init.constant_(m.weight,1)
                init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                init.normal_(m.weight,std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias,0)
    def forward(self,enc_out,word):
        '''
        :param enc_out:
        :param word:
        :return:
        '''
        attention_weighted_encoding,alpha=self.atten(enc_out,word)  #[batch_size,512]
        res=torch.cat([attention_weighted_encoding,word],dim=1)
        g=self.sigmoid(self.relu(self.fc_sig(res)))
        i=self.relu(self.fc(res))
        aoa_atten=g*i
        return aoa_atten,alpha
# if __name__=='__main__':
#     #测试 通道注意力和空间注意力
#     # vis=torch.randn((3,512,14,14))    #(batch_size,num_pixels,d_model)
#     # inf=torch.randn((3,512,14,14))
#     # #d_model,d_k,d_v,head,reduction=16,drop=0.5
#     # self_channel_atten=self_Channel_Atten(d_model=512,d_k=512,d_v=512,head=8,drop=0.5)
#     # y=self_channel_atten(vis,inf)
#     # print(y.shape)
#     word=torch.randn((5,512))
#     enc_out=torch.randn((5,196,1024))
#     atten=Aoa(encoder_dim=1024,decoder_dim=512,attention_dim=512,dropout=0.5)
#     attention_weighted_encoding, alpha=atten(enc_out,word)
#     print(attention_weighted_encoding.shape)
#     print(alpha.shape)



