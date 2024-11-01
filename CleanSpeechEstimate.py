import torch
import torch.nn as nn

class CleanSpeechEstimation(nn.Module):
    def __init__(self, power_law=0.3):
        super(CleanSpeechEstimation, self).__init__()
        self.power_law = power_law

    def decompressed_result(self,speech_i,speech_r,a):
        S_i = torch.sign(speech_i) * torch.pow(speech_r, 1 / a)
        S_r = torch.sign(speech_r) * torch.pow(speech_r, 1 / a)
            
        S = torch.complex(S_i,S_r)
        return S


    def forward(self, complex_mask,compressed_real,compressed_img):
        # Estimated clean speech signal
        a = 0.6
        imag_mask = complex_mask[0,:,:]
        # print("Imaginary M_i",imag_mask.shape)

        real_mask = complex_mask[1,:,:]
        # print("Real M_r",real_mask.shape)

        # print("Compressed Imag shape",compressed_img.shape)
        # print("Compressed Real shape",compressed_real.shape)

        # print(compressed_img)
        # print(compressed_real)

        # print("Is it contain NAN at compressed_real after bands output",torch.isnan(compressed_real).any()) 

        real_mask = real_mask.permute(1, 0).unsqueeze(0).expand(1,626,257)
        imag_mask = imag_mask.permute(1, 0).unsqueeze(0).expand(1,626,257)

        real_mask = real_mask.permute(0,2,1)
        imag_mask = imag_mask.permute(0,2,1)
        
        # print("Shape of compressed_real:", compressed_real.shape)
        # print("Shape of real_mask:", real_mask.shape)
        # print("Shape of compressed_img:", compressed_img.shape)
        # print("Shape of imag_mask:", imag_mask.shape)


        # estimated_complex_speech = (real_mask + 1j * imag_mask) * noisy_signal
        S_r = compressed_real * real_mask - compressed_img* imag_mask
        S_i = compressed_real * real_mask + compressed_img * imag_mask
        
        # print("Is it contain NAN at S_i after bands output",torch.isnan(S_i).any()) 
        # print("Is it contain NAN at S_r after bands output",torch.isnan(S_r).any()) 

        # estimated_complex_speech = (S_r + 1j * S_i) * noisy_signal/

        Speech = self.decompressed_result(S_r,S_i,a)
        
                
        return Speech
