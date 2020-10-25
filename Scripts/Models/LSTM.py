import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_first =True




class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size ,num_classes , rnndropout):
        super().__init__()
        self.hidden_size = hidden_size

        self.drop = nn.Dropout(rnndropout)
        self.fc = nn.Linear(hidden_size, num_classes) 
        self.rnn = nn.LSTM(input_size,hidden_size,batch_first=True)



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



