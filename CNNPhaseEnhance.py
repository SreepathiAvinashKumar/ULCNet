import torch
import torchaudio
import torch.nn as nn


class CNNPhaseEnhancement(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=(1, 3), stride=1, padding=1):
        super(CNNPhaseEnhancement, self).__init__()
        
        # Intermediate Feature Computation (Noisy Phase and Intermediate Real Magnitude Mask)
        # self.combine_features = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # CNN Processing
        self.conv1 = nn.Conv2d(257,257, kernel_size, stride, padding)
        self.bn1 = nn.LayerNorm(257,eps=1e-5)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(257, 257, kernel_size, stride, padding)
        self.bn2 = nn.LayerNorm(257,eps=1e-5)

        # Complex Mask Estimation - Pointwise Conv
        self.complex_mask_estimation_pointwise = nn.Conv2d(6, 2, kernel_size=1)  # 2 for real and imaginary parts # Pointwise conv
        self.bn3 = nn.LayerNorm(2,eps=1e-5)

    def intermediate_feature_computation(self,x,noisy_phase):
        # print("Is it contain NAN at Nois Phase after bands output",torch.isnan(noisy_phase).any()) 
        y_r = x * torch.cos(noisy_phase)
        y_i = x * torch.sin(noisy_phase)
        print("Is it contain NAN at y_r after bands output",torch.isnan(y_r).any()) 
        print("Is it contain NAN at y_i after bands output",torch.isnan(y_i).any()) 

        return y_r,y_i

    def forward(self, magnitude_mask, compressed_phase):

        y_r,y_i = self.intermediate_feature_computation(magnitude_mask,compressed_phase)
        # print("y_r dimension is ",y_r.shape)


        # print("y_i dimension is ",y_r.shape)

        x = torch.cat((y_r, y_i), dim=0)
        print("Is it contain NAN at After Contcat after bands output",torch.isnan(x).any()) 
        print("After the y_r and y_i concat",x.shape)

        # print("before reshape",x) # [2, 257, 626]

        x = x.permute(1,0,2)
        # print("after reshape",x.shape)

        # CNN Processing
        x = self.conv1(x)
        x = x.permute(2,1,0) # size[257, 4, 626]

        x = self.bn1(x)
        print("Is it contain NAN at CRN2 and batchn1 after bands output",torch.isnan(x).any()) 

        x = x.permute(1,2,0)
        x = x.permute(1,0,2)

        x = self.relu(x)
        x = self.conv2(x)
        x = x.permute(1,2,0) 

        x = self.bn2(x)
        x = x.permute(1,0,2)


        # print("After conv 2 layers in CNNPhase",x.shape)
        x = x.permute(1,0,2)
        # print("After reshape again",x.shape)
        
        # # Complex Mask Estimation
        complex_mask = self.complex_mask_estimation_pointwise(x)

        # print(complex_mask.shape)
        complex_mask = complex_mask.permute(1,0,2)
        complex_mask = complex_mask.permute(0,2,1)
        complex_mask = self.bn3(complex_mask)
        complex_mask = complex_mask.permute(1,0,2)
       
        print("Is it contain NAN at complex_mask and batchn1 after bands output",torch.isnan(complex_mask).any()) 
        complex_mask = complex_mask.permute(2,0,1)

        # print("Complex mask",complex_mask.shape)
        # print("Imaginary Mask",complex_mask[0,:,:].shape)
        return complex_mask
    

