import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter

from typing import *
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_first =True



from enum import IntEnum
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
 


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

        M=torch.zeros(batchSize , self.input_sz).double()

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
        # Set initial states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        h0 = h0.double()
        c0 = c0.double()
        x = self.drop(x)
        output, _ = self.rnn(x, (h0, c0))
        output = self.drop(output)
        output=output[:,-1,:]
        out = self.fc(output)
        out =F.softmax(out, dim=1)
        return out



class InputCellAttentionMatrix(nn.Module):

    def __init__(self, input_sz: int, hidden_sz: int,r:int,d_a:int):
        super().__init__()
        self.r=r
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.weight_ih = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.weight_iBarh = Parameter(torch.Tensor(r* input_sz,  hidden_sz* 4))
        self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = Parameter(torch.Tensor(hidden_sz * 4))
        self.r=r
        self.softmax_=nn.Softmax()
        self.linear_first = torch.nn.Linear(input_sz,d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)
        self.init_weights()
        # print("LSTMWithCellAttentionMatrix")
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
        #print(matrixM.shape)

        return matrixM

    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
 
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied
 
        Returns:
            softmaxed tensors
 
        """
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

        M=torch.zeros(batchSize , self.r , self.input_sz).double()

        for t in range(seq_sz):
            x_t = x[:, t, :]
            if(t==0):
                H=x[:, 0, :].view(batchSize,1,self.input_sz)

                M = self.getMatrixM(H)
            elif(t>0):
                H=x[:, :t+1, :]

                M = self.getMatrixM(H)

            xBar=M.view(batchSize,self.r*self.input_sz)
            gates = xBar @ self.weight_iBarh + h_t @ self.weight_hh + self.bias

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



class LSTMWithInputCellAttention_Unoptimized(nn.Module):
    def __init__(self, input_size, hidden_size ,num_classes , rnndropout ,r,d_a):
        super().__init__()
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(rnndropout)  
        self.fc = nn.Linear(hidden_size, num_classes) 
        self.rnn =InputCellAttentionMatrix(input_size, hidden_size,r,d_a)



        
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        h0 = h0.double()
        c0 = c0.double()
        x = self.drop(x)
        output, _ = self.rnn(x, (h0, c0))
        output = self.drop(output)
        output=output[:,-1,:]
        out = self.fc(output)
        out =F.softmax(out, dim=1)
        return out