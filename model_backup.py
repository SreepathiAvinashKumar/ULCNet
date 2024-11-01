import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from CustomDataLoader import MyDataset
from torch.utils.data import DataLoader
from CRN2 import CRNBasedMagnitudeEstimation
from CNNPhaseEnhance import CNNPhaseEnhancement
from CleanSpeechEstimate import CleanSpeechEstimation
import torch.optim.lr_scheduler as lr_scheduler
from torchsummary import summary

# There we get Channel Dimension only if we get the complex numbers as individual values, otheriwse if we get as complex_value = True then all are as 1 on channel dimension

class SpeechEnhancementModel(nn.Module):
    def __init__(self, crn_params, cnn_params, power_law):
        super(SpeechEnhancementModel, self).__init__()
        self.power_law = power_law
        self.magnitude_estimation = CRNBasedMagnitudeEstimation(**crn_params)
        self.phase_enhancement = CNNPhaseEnhancement(**cnn_params)
        self.clean_speech_estimation = CleanSpeechEstimation(power_law)

    def stft(self,x, fft_size=512, hop_size=256, window='hann'): 
    # Create window function
        if window == 'hann':
            window = torch.hann_window(fft_size)
        else:
            raise ValueError("only hann window is supported.")

        # Compute STFT
        stft_matrix = torch.stft(x, fft_size, hop_size, window=window,return_complex=True) #  = false
        print(stft_matrix)
        return torch.view_as_real(stft_matrix)

    def power_law_compression(self, signal):
        real= signal[..., 0] # real part
        imag= signal[..., 1] # imag part

        compressed_real = torch.pow(torch.abs(real), self.power_law) * torch.sign(real)
     
        compressed_img = torch.pow(torch.abs(imag), self.power_law) * torch.sign(imag)

        compressed_mag = torch.sqrt((compressed_real)**2 + (compressed_img) **2)

        small_value = 1e-7
        compressed_real = torch.where(compressed_real == 0, small_value, compressed_real)
        compressed_img = torch.where(compressed_img == 0, small_value, compressed_img)

        compressed_phase = torch.arctan(compressed_img/compressed_real)
        print("Is it contain NAN at Compressed after bands output",torch.isnan(compressed_phase).any()) 
        # print("compressed phase",compressed_phase)
        
        return compressed_mag,compressed_phase,compressed_real,compressed_img
    

    def istft(self, stft_matrix, fft_size=512, hop_size=256, window='hann'):
    # Convert the input back to complex form if necessary
        # if stft_matrix.size(-1) == 2:
            # stft_matrix = torch.view_as_complex(stft_matrix)

        # Create window function
        if window == 'hann':
            window = torch.hann_window(fft_size)
        else:
            raise ValueError("only hann window is supported.")

        # Compute iSTFT
        x_reconstructed = torch.istft(stft_matrix, n_fft=fft_size, hop_length=hop_size, window=window)
    
        return x_reconstructed
    
    def pad_signal(self,signal, target_length):
        k = target_length
        padding = k - signal.size(-1)
        # print("target is ",k)
        # print("signal size is ",signal.size(-1))
        # print("padding is ",padding)
        if padding > 0:
            signal = torch.nn.functional.pad(signal, (0, padding))
            # print("signal shape is ",signal.shape)
        return signal

    def forward(self, noisy_signal):
        print(noisy_signal.shape)
        # Apply stft 
        stft_matrix = self.stft(noisy_signal)

        # print("After STFT :",stft_matrix.shape)

        # Apply power law compression
        compressed_mag,compressed_phase,compressed_real,compressed_img = self.power_law_compression(stft_matrix)
        # print("comp phase shape",compressed_phase.shape)

        # print("After power law compression of mag",compressed_mag.shape)

        print("------ CRN Mag estimation block----------")


        magnitude_mask = self.magnitude_estimation(compressed_mag)

        # print("Is it contain NAN at Magnitude Mask after bands output",torch.isnan(magnitude_mask).any()) 
        # print("Is it contain NAN at Compressed after bands output",torch.isnan(compressed_phase).any()) 

        # print("Magnitude Mask",magnitude_mask)
        # print("Magnitude Mask shape",magnitude_mask.shape)


        print("---- CNN based phase enhancement--------")

        # Stage 2: CNN-based Phase Enhancement

        complex_mask = self.phase_enhancement(magnitude_mask, compressed_phase)
        
        # Clean Speech Estimation
        estimated_clean_speech = self.clean_speech_estimation(complex_mask,compressed_mag,compressed_phase)

        speech = self.istft(estimated_clean_speech)

        # print("Speech in time domain",speech.shape)

        target_length = noisy_signal.size(-1)
        # speech = self.pad_signal(speech,target_length)
        # print("AFter Padding in time domain",speech.shape)
        
        return speech

# Example usage
crn_params = {
    'in_channels': 257,
    'out_channels': 32, 
    'hidden_size': 64, 
    'num_subbands': 2,
}
cnn_params = {
    'in_channels': 2, 
    'out_channels': 32
}


csv_file = './DataLoad/files.csv'
train = './DataLoad/train'
label = './DataLoad/label'  
batches = 1


mdataset = MyDataset(csv=csv_file,train_dir=train,label_dir=label)

train_set ,test_set  = torch.utils.data.random_split(mdataset,[23,23])

train_loader = DataLoader(train_set,batch_size=batches,shuffle=True)
test_loader = DataLoader(test_set,batch_size=batches,shuffle=True)


sample_rate = 16000
duration = 10
channels = 2
num_epochs = 1
learning_rate = 0.00004
power_law = 0.6

# Initialize the model, loss function, and optimizer
model = SpeechEnhancementModel(crn_params, cnn_params,power_law)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=10)


for epoch in range(num_epochs):
    running_loss = 0.0
    for i,data in enumerate(train_loader):
        noisy_waveform, clean_waveform = data

        # Forward pass
        outputs = model(noisy_waveform)
        # model.eval()
        
        # Compute loss
        loss = criterion(outputs, clean_waveform)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()

        print('=======================================================================')
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
        print('=======================================================================')


print("Training finished!")

# summary(model,(1,1360000))
# k = model(torch.tensor(mdataset.__getitem__(1)[0]))