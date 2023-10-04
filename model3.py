import torch
import torch.nn as nn
import torch.nn.init as init

import torch.nn as nn
import torch.nn.init as init

import torch.nn as nn
import torch.nn.init as init

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(input_size + hidden_size, 1)

    def forward(self, encoder_outputs, hidden_state):
        seq_len = encoder_outputs.size(0)
        h = hidden_state.repeat(seq_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attention_weights(torch.cat([encoder_outputs, h], dim=2)))
        attention_weights = nn.functional.softmax(energy, dim=0)
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=0)
        return context_vector.unsqueeze(0)

import torch.nn as nn
import torch.nn.init as init

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(input_size + hidden_size, 1)

    def forward(self, encoder_outputs, hidden_state):
        seq_len = encoder_outputs.size(0)
        h = hidden_state.repeat(seq_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attention_weights(torch.cat([encoder_outputs, h], dim=2)))
        attention_weights = nn.functional.softmax(energy, dim=0)
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=0)
        return context_vector.unsqueeze(0)

class CRNN(nn.Module):
    def __init__(self, num_classes, drop_out_rate = 0.35):
        super().__init__()
        #CNN
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2,2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same', bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=(2,2))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding='same', bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding='same', bias=True),
            nn.Dropout(drop_out_rate),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding='same', bias=True),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same', bias=True),
            nn.Dropout(drop_out_rate),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))     
        )   
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=2, padding='same', bias=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU()
        )
        
        # Additional convolutional layers
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same', dilation=2, bias=True),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(),
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same', dilation=2, bias=True),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(),
        )
        
        # Residual connections
        self.residual1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_features=512)
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, bias=True),
            nn.BatchNorm2d(num_features=512)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU()
        )

        #RNN
        self.rnn1 = nn.LSTM(input_size=128, hidden_size=256, bidirectional=True, batch_first=True) 
        self.rnn2 = nn.LSTM(input_size=512, hidden_size=512, bidirectional=True, batch_first=True)
        
        # Attention mechanism
        self.attention = Attention(512, 512)
        
        #FC
        self.fc2 = nn.Linear(1024, num_classes)
        
        #Softmax
        self.softmax = nn.LogSoftmax(dim=2)
        
        #He/Kaming weight initialization
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        
        # Additional convolutional layers with dilated convolutions
        x = self.conv8(x)
        x = self.conv9(x)
        
        # Residual connections
        residual = self.residual1(x)
        x = self.conv9(x) + residual
        residual = self.residual2(x)
        x = self.conv9(x) + residual
        
        #output of cnn layer has shape (batchsize, channel, height, width)
        
        # input to an LSTM layer in PyTorch has shape (sequence_length, batch_size, input_size) and width should be sq_len (chữ cái thường dễ nhận diện hơn khi chia dọc từng đoạn hơn là chia ngang)
        # CNN to RNN
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = self.fc1(x)

        # Attention mechanism
        encoder_outputs = x.permute(1, 0, 2)
        hidden_state = torch.zeros(2, x.size(0), 512)
        if torch.cuda.is_available():
            hidden_state = hidden_state.cuda()
        context_vector = self.attention(encoder_outputs, hidden_state)
        x = torch.cat([context_vector, x], dim=2)
        
        x = self.rnn1(x)[0]
        x = self.rnn2(x)[0]
        x = self.fc2(x)
        x = self.softmax(x)
        return x.view(x.size(1), x.size(0), -1)
        

    
if __name__ == '__main__':    
    input_data = torch.rand(8, 1, 64, 128)
    
    model = CRNN(num_classes=188).cuda()
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    while True:
        result = model(input_data)
        print(result.shape)
        break