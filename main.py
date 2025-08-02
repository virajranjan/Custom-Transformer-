
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


print(input_embedding(2,5).embedding)