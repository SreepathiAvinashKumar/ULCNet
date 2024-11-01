import torch
import torchaudio
import torch.nn as nn


class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DepthwiseSeparableConv1D, self).__init__()
    
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.bn3 =  nn.LayerNorm(in_channels,eps=1e-5)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn4 =  nn.LayerNorm(out_channels,eps=1e-5)
        self.relu = nn.ReLU()
        
    def forward(self, x): # no Nan values
        x = self.depthwise(x)
        x = x.permute(0,2,1)
        x = self.bn3(x)
        x = x.permute(0,2,1)
        x = self.pointwise(x)
        x = x.permute(0,2,1)
        x = self.bn4(x)
        x = x.permute(0,2,1)
        x = self.relu(x)
        return x
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,index,kernel_size=(3), stride=1,padding=1):
        super(ConvBlock, self).__init__()
        self.conv = DepthwiseSeparableConv1D(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.index = index
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.bn4 = nn.LayerNorm(out_channels)


    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0,2,1)
        x = self.bn4(x)
        x = x.permute(0,2,1)
        x = self.relu(x)
        if self.index > 0:
            x = x.permute(0,2,1)
            x = self.pool(x)
            x = x.permute(0,2,1)
        return x


class FGRU(nn.Module): # return frequency bands as 128 
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True):
        super(FGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,bidirectional=bidirectional)
        self.bn_gru = nn.LayerNorm(hidden_size * 2)


    def forward(self, x):
        x, _ = self.gru(x)
        # print(torch.isnan(x).any())
        # print(x)
        # x = x.permute(0,2,1)
        x = self.bn_gru(x)
        # print("after",torch.isnan(x).any())
        # x = x.permute(0,2,1)
        return x
    

class SubbandGRUBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SubbandGRUBlock, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size)
        self.bn_gru = nn.LayerNorm(hidden_size)
        
        
    def forward(self, x):
        # print("At the SubbandGRU",x.shape)
        output, _ = self.gru1(x)
        # output = output.permute(0,2,1)
        # print("after the permute",output.shape)
        output = self.bn_gru(output)
        # print("after the layer norm shape",output.shape)
        # output = output.permute(0,2,1)
        # print("again permute perfomed shape",output.shape)

        # output, _ = self.gru1(output)
        # print("After GRU", output.shape)
        print("shape of output",output.shape)
        return output


class TemporalGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128,num_subbands=2):
        super(TemporalGRU, self).__init__()
        self.num_subbands = num_subbands
        self.bn3 = nn.LayerNorm(hidden_size,eps=1e-4)
        self.subband_blocks = nn.ModuleList([
            SubbandGRUBlock(input_size, hidden_size) for _ in range(num_subbands)
        ])
  
    def subband_processing(self,subband_inputs):
        subband_outputs = []
        for i in range(self.num_subbands):
            subband_output = self.subband_blocks[i](subband_inputs[i])
            subband_outputs.append(subband_output)

        combined_output = torch.cat(subband_outputs, dim=-1)
        # combined_output = self.bn3(combined_output)

        print("After combined output ",combined_output.shape)
        return combined_output
    
    def decompose_subbands(self,x, num_subbands): # ([1, 64, 5168])
            # spliting the input into equal parts
        # print("Number of subbands",num_subbands)
        x = x.permute(0, 2, 1)  # ([1,5168,64])
        subband_length = x.size(-1) // num_subbands # 32 
        subbands = [x[:,:,i*subband_length:(i+1)*subband_length] for i in range(num_subbands)] # ([1, 5169,64]) x 2
        # print(subbands[0].shape)
        return subbands

    def forward(self, x):
        subband_inputs = self.decompose_subbands(x, self.num_subbands)
        processed_ouput = self.subband_processing(subband_inputs)
        return processed_ouput
        # x, _ = self.gru(x)
        # return x
    

class CRNBasedMagnitudeEstimation(nn.Module):
    def __init__(self, in_channels, out_channels,hidden_size,num_subbands,kernel_size=(1,3), stride=1, padding=1):
        super(CRNBasedMagnitudeEstimation, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_subbands = num_subbands
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Frequency-axis Bidirectional GRU
        self.fgru = FGRU(80,self.hidden_size)

        self.temporal_gru = TemporalGRU(32, 128, self.num_subbands)

        # Point-wise Convolution
        self.pointwise_conv = nn.Conv1d(128, 64, kernel_size=1)
        self.bn_pointwise = nn.BatchNorm1d(64)
        self.bn_2= nn.BatchNorm1d(80)

        self.fc1 = nn.Linear(256, 257)
        # self.bn_fc1 = nn.BatchNorm1d(257)
        self.fc2 = nn.Linear(257, 257)
        # self.bn_fc2 = nn.BatchNorm1d(257)
        
    def perform_conv(self,x): # Not used where corresponding function directly implemented in forward function
        channels = [32, 64, 96, 128]
        for i,out_channels in enumerate(channels):
            
            conv_block = ConvBlock(in_channel, out_channels,i)
            x = conv_block(x,i)
            if i > 1:
                x = self.pool(x)
            print("Each Band",x.shape)
            in_channel = out_channels
        print(in_channel)
        in_channel = self.in_channels

        return x
        
    def forward(self, x):
        
        #Channelwise feature reorientation 

        limit = [0,16, 32, 64, 128, 257]

        filters = [64,64,64,32,32]
        subbands = []
        conv = len(filters)
        i = 0
        # subband splitting 
        
        while i < len(limit) - 1:
                start = limit[i] +1
                end = limit[i + 1]
                # print(start,end)
                subband = x[:, start:end, :]
                i = i +1
                subbands.append(subband)

        # Applying Conv to those sub splittings bands 

        # print("each subband has dim",subbands[0].shape)
        
        bands_output = []
        for band in subbands:
            for j in range(0,conv):
                # print(band.shape[1],filters[j]) # not neccassary to print 
                band = ConvBlock(band.shape[1],filters[j],j)(band)
                # print("band shape is ",band.shape)
            # print('==========')
            bands_output.append(band)
        
        # print("after suband splitting and applying conv",bands_output)
        
        x = torch.cat(bands_output, dim=1)
        x = self.bn_2(x)

        print("Is it contain NAN at CRN2 after bands output",torch.isnan(x).any()) 
        
        print("After concatnation of conv subband channels",x.shape)

        x = x.permute(0,2,1) #  
        # print("After permute",x.shape)

        x = self.fgru(x)
        # print("at fgru",x)
        # print("After fgru",x.shape)

        x = x.permute(0,2,1)  
        # print("After permute", x.shape) 

        x = self.pointwise_conv(x) 
        x = self.bn_pointwise(x)
        # print("After pointwise", x.shape) 

        x =  self.temporal_gru(x)
        # print("After temporal gru", x.shape)

        x = self.fc1(x)
        # x = self.bn_fc1(x)
        # print("After fc1", x.shape)
        x = self.fc2(x)
        # x = self.bn_fc2(x)
        # print("After fc2", x.shape)

        # print("final output after view ",x.shape)
        x = x.permute(0,2,1) 
        # print("after permute",x.shape)
        # print("at before return",x)
        return x

    


    
