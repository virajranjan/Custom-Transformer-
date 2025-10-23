
import torch 
import torch.nn as nn
import math

class input_embedding(nn.Module):
    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size =vocab_size
        self.embedding=nn.Embedding(d_model,vocab_size)
    def forward(self, X):
        return self.embedding(X)* math.sqrt(self.d_model)
    
class positonal_encoding(nn.Module):

    def __init__(self,d_model:int ,seq_len:int, dropout:int):
        super().__init__()
        self.d_model= d_model
        self.seq_len = seq_len
        self.dropout = dropout
        
        pe = torch.zeros(seq_len,d_model)
        position = torch.arange(0,seq_len, dtype = float).squeeze(1)
        div =torch.exp( torch.arange(0,d_model,2).float()*(-math.log(10000)/d_model))

        pe[:,::2]=torch.sin(position*div)
        pe[:,1::2]=torch.cos(position*div)
        pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, X):
        X=X*(self.pe[:,:X.shape[1],:]).requires_grad(False)

class Layer_Normalization(nn.Module):
    def __init__(self, ep:float =10**-6,)->None:
        super().__init__()
        self.ep = ep
        self.alpha = nn.Parameter(torch.ones(1)) # multiplied
        self.betta = nn.Parameter(torch.zeros(1)) # added
    def forward(self ,X):
        mean = X.mean(dim = -1,keepdim = True)
        dav = X.std(dim =-1,keepdim = True)
        return self.alpha*(X-mean)/(dav+self.ep)+self.betta 
    

class Feed_forward(nn.Module):
    def __init__(self,d_model:int, d_ff:int, dropout:float):
        super().__init__()
        self.layer1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(d_ff,d_model)        
    
    def forward(self,x):
        return self.layer2(self.dropout(torch.relu(self.layer1(x))))
    
# Multi-head attention  
class MultiheadAttentionBlock(nn.Module):

    def __init__(self,d_model:int, dropout:float, h:int):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert  d_model%h ==0, "d_model is not divisible by h"

        self.d_k = d_model//h
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv

        self.w_o = nn.Linear(d_model, d_model) #W0
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value,mask, dropout:nn.Dropout):
        d_k = query.shape[-1]
        #(batch,h,seq_len, seq_len)-->(batch, h, seq_len, seq_len)
        attention_score = (query@ key.tanspose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_score.masked_fill_(mask == 0,-1e9)
        attention_score = attention_score.softmax(dim = -1) # (batch, h, seq_len , seq_len)\
        if dropout is not None:
            attention_score= dropout(attention_score)
        return attention_score@value , attention_score

    def forward(self, q , k , v ,mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query  = query.view(query.shape[0],query.shape[1],self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0],key.shape[1],self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0],value.shape[1],self.h,self.d_k).transpose(1,2)

        x, self.attentiom_score = MultiheadAttentionBlock.attention(query,key, value, mask , self.dropout)

        # (batch,j,seq_len, d_k)--> (batch, seq_len, h, d_k) --> (batch, h, seq_len , d_k)
        x = x.transpose(1,2).contigous().view(x.shape[0],-1,self.h*self.d_k)
        #(batch, seq_len , d_model) --> (batch, seq_len, d_model)

        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout:float)->None:
        super().__init()
        self.dropout = nn.Dropout(dropout)
        self. norm = Layer_Normalization()
    def forward(self, x, sublayer):
        return x+self.dropout(sublayer(self.norm(x)))


# Encode block 
class Encoderblock(nn.Module):
    def __init__(self, self_attention_block:MultiheadAttentionBlock, feed_forward_block:Feed_forward, dropout:float)-> None:
        super().__init__()
        self .self_attention_block = self_attention_block
        self. feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x,lambda x:self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connection[1](x,self.feed_forward_block)
        return x
    
# single tansformer can have muiltiple encoder block, we will define N Encoder.
class Encoder(nn.Module):
     def __init__(self, layers:nn.Module)->None:
         super().__init__()
         self.layer = layers
         self.norm = Layer_Normalization()
    
     def forward(self,x,mask):
         for layer  in self.layers:
              x= layer(x,mask)
         return self.norm(x)

# implementing the decoder block 
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block :MultiheadAttentionBlock, cross_attention_block:MultiheadAttentionBlock,feed_forward_block:Feed_forward, dropout:float)->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = self.cross_attention_block
        self.feed_forward_block = self.feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
    
    def forward(self,x, encoder_output,src_mask,tgt_mask):
        x = self.residual_connection[0](x,lambda x:self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connection[1](x, lambda x:self.cross_attention_block(x,encoder_output,encoder_output,src_mask))
        x = self.residual_connection[2](x,self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers:nn.Module)->None:
        super().__init__()
        self.layers = layers
        self.norm = Layer_Normalization

        def forward(self, x, encoder_output, src_mask, tgt_mask):
            for layer in self.layer:
                x = layer(x,encoder_output, src_mask, tgt_mask)
            return self.norm(x)

class Projection_layer(nn.Module):
    def __init__(self, d_model , vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (batch_size, seq_len, d_model)--> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x),dim = -1)
