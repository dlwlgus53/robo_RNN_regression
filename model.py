'''
adopted from pytorch.org (Classifying names with a character-level RNN-Sean Robertson)
'''

import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

        self.batch_size = batch_size
        self.hidden_size = hidden_size

        
    def forward(self, input):
        '''
        input = input.unsqueeze(0)
        '''
        #import pdb; pdb.set_trace()
        hidden = self.initHidden()
        hidden = (hidden[0],hidden[1])
        output, hidden = self.rnn(input, hidden)
                
        output = self.fc(output) 

        return output

    def initHidden(self):
        return torch.zeros(2, 1, self.batch_size, self.hidden_size).to('cuda')
