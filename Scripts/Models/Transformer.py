import torch
import torch.nn as nn 
import copy
import math
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def creatMask(batch,sequence_length):
    mask = torch.zeros(batch,sequence_length,sequence_length)
    for i in range (sequence_length):
        mask[:,i,:i+1]=1
    return mask


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None,returnWeights=False):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    # print("Scores in attention itself",torch.sum(scores))
    if(returnWeights):
        return output,scores

    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None,returnWeights=False):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next

        if(returnWeights):
            scores,weights = attention(q, k, v, self.d_k, mask, self.dropout,returnWeights=returnWeights)
            # print("scores",scores.shape,"weights",weights.shape)
        else:
            scores = attention(q, k, v, self.d_k, mask, self.dropout)

        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
        # print("Attention output", output.shape,torch.min(output))
        if(returnWeights):
            return output,weights
        else:
            return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=400, dropout = 0.1):
        super().__init__() 
    
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x




class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x,mask=None,returnWeights=False):
        x2 = self.norm_1(x)
        # print(x2[0,0,0])
        # print("attention input.shape",x2.shape)
        if(returnWeights):
            attenOutput,attenWeights= self.attn(x2,x2,x2,mask,returnWeights=returnWeights)
        else:
            attenOutput= self.attn(x2,x2,x2,mask)
        # print("attenOutput",attenOutput.shape)
        x = x + self.dropout_1(attenOutput)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        if(returnWeights):
            return x,attenWeights
        else:
            return x




class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 100, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)

        pe = Variable(self.pe[:,:seq_len], requires_grad=False)

        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, input_size,seq_len, N, heads, dropout):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(input_size,seq_len, dropout=dropout)
        self.layers = get_clones(EncoderLayer(input_size, heads, dropout), N)
        self.norm = Norm(input_size)
    def forward(self, x,mask = None, returnWeights=False):
        x = self.pe(x)

        for i in range(self.N):
            if(i==0 and returnWeights):
                x,weights = self.layers[i](x,mask=mask,returnWeights=returnWeights)
            else:
                # print(i)
                x = self.layers[i](x,mask=mask)


        if(returnWeights):
            return self.norm(x),weights
        else:
            return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, input_size,seq_len , N, heads, dropout,num_classes,time=100):
        super().__init__()
        self.encoder = Encoder(input_size,seq_len, N, heads, dropout)
        self.out = nn.Linear(input_size, num_classes) 
        self.tempmaxpool = nn.MaxPool1d(time)
    def forward(self, src,returnWeights=False):
        mask= creatMask(src.shape[0],src.shape[1]).to(device)
        # print(src.shape)
        if(returnWeights):
            e_outputs,weights,z = self.encoder(src,mask,returnWeights=returnWeights)
        else:
            e_outputs = self.encoder(src,mask)

        e_outputs=self.tempmaxpool(e_outputs.transpose(1, 2)).squeeze(-1)
        output = self.out(e_outputs)
        output =F.softmax(output, dim=1)
        if(returnWeights):
            return output,weights
        else:
            return output

    



