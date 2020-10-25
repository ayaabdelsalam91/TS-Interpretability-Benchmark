import torch.nn.functional as F
from torch import nn
from typing import *
from torch.nn import Parameter
from torch.autograd import Variable
import copy
import math
import torch
from torch.nn.utils import weight_norm
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_first =True

from enum import IntEnum
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
 



class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size ,num_classes , rnndropout):
        super().__init__()
        self.hidden_size = hidden_size

        self.drop = nn.Dropout(rnndropout)
        self.fc = nn.Linear(hidden_size, num_classes) 
        self.rnn = nn.LSTM(input_size,hidden_size,batch_first=True)



    def forward(self, x):
        x= x.transpose(1, 2)
        # Set initial states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        x = self.drop(x)
        output, _ = self.rnn(x, (h0, c0))
        output = self.drop(output)
        output=output[:,-1,:]
        out = self.fc(output)
        return F.log_softmax(out, dim=1)






class InputCellAttention(nn.Module):

    def __init__(self, input_sz: int, hidden_sz: int,r:int,d_a:int):
        super().__init__()
        self.r=r
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.weight_iBarh = Parameter(torch.Tensor(input_sz,  hidden_sz* 4))
        self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = Parameter(torch.Tensor(hidden_sz * 4))
        self.r=r
        self.linear_first = torch.nn.Linear(input_sz,d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)


    def getMatrixM(self, pastTimeSteps):

        x= self.linear_first(pastTimeSteps)

        x = torch.tanh(x)
        x = self.linear_second(x) 
        x = self.softmax(x,1)
        attention = x.transpose(1,2) 
        matrixM = attention@pastTimeSteps 
        matrixM = torch.sum(matrixM,1)/self.r

        return matrixM

    def softmax(self,input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])

        soft_max_2d = F.softmax(input_2d, dim=-1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)


    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor]]=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size
        batchSize=x[:, 0, :].size()[0]

        M=torch.zeros(batchSize , self.input_sz)

        for t in range(seq_sz):
            x_t = x[:, t, :]
            if(t==0):
                H=x[:, 0, :].view(batchSize,1,self.input_sz)

                M = self.getMatrixM(H)
            elif(t>0):
                H=x[:, :t+1, :]

                M = self.getMatrixM(H)


            gates = M @ self.weight_iBarh + h_t @ self.weight_hh + self.bias

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:,:, :HS]), # input
                torch.sigmoid(gates[:,:, HS:HS*2]), # forget
                torch.tanh(gates[:,:, HS*2:HS*3]),
                torch.sigmoid(gates[:,:, HS*3:]), # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        hidden_seq = hidden_seq.squeeze(1)

        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)





class LSTMWithInputCellAttention(nn.Module):
    def __init__(self, input_size, hidden_size ,num_classes , rnndropout ,r,d_a):
        super().__init__()
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(rnndropout)  
        self.fc = nn.Linear(hidden_size, num_classes) 
        self.rnn =InputCellAttention(input_size, hidden_size,r,d_a)



        
    def forward(self, x):
        x= x.transpose(1, 2)
        # Set initial states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)

        x = self.drop(x)
        output, _ = self.rnn(x, (h0, c0))
        output = self.drop(output)
        output=output[:,-1,:]
        out = self.fc(output)
        return F.log_softmax(out, dim=1)







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
        
    def forward(self, x,returnWeights=False):
        x2 = self.norm_1(x)
        # print(x2[0,0,0])
        # print("attention input.shape",x2.shape)
        if(returnWeights):
            attenOutput,attenWeights= self.attn(x2,x2,x2,returnWeights=returnWeights)
        else:
            attenOutput= self.attn(x2,x2,x2)
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
    def forward(self, x,returnWeights=False):
        x = self.pe(x)

        for i in range(self.N):
            if(i==0 and returnWeights):
                x,weights = self.layers[i](x,returnWeights=returnWeights)
            else:
                # print(i)
                x = self.layers[i](x)


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
        src= src.transpose(1, 2)
        if(returnWeights):
            e_outputs,weights = self.encoder(src,returnWeights=returnWeights)
        else:
            e_outputs = self.encoder(src)

        e_outputs=self.tempmaxpool(e_outputs.transpose(1, 2)).squeeze(-1)
        output = self.out(e_outputs)
        output = F.log_softmax(output, dim=1)
        if(returnWeights):
            return output,weights
        else:
            return output

    



class RT(nn.Module):
    def __init__(self, input_size, d_model, output_size, h, rnn_type, ksize, n_level, n, dropout=0.2, emb_dropout=0.2):
        super(RT, self).__init__()
        self.encoder = nn.Linear(input_size, d_model)
        self.rt = RTransformer(d_model, rnn_type, ksize, n_level, n, h, dropout)
        self.linear = nn.Linear(d_model, output_size)

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = x.transpose(-2,-1)
        x = self.encoder(x)
        x = self.rt(x)  # input should have dimension (N, C, L)
        x = x.transpose(-2,-1)
        o = self.linear(x[:, :, -1])
        return F.log_softmax(o, dim=1)

class CNN(nn.Module):
    def __init__(self, output_size, h, dropout1=0.25, dropout2=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(dropout1)
        self.dropout2 = nn.Dropout2d(dropout2)
        self.fc1 = nn.Linear(9216, h)
        self.fc2 = nn.Linear(h, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


