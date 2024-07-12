#!/usr/bin/env python
# coding: utf-8

# # Research notebook for backdoor attacks on audio generative diffusion models

# The diffusion model used in this notebook was based on a model from an assignment for week 11 of the 2023 Deep Learning course (NWI-IMC070) of the Radboud University.

# # Here is the initial code:

import torchaudio
import torchvision
from d2l import torch as d2l
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import os
import joblib


if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')
print(device)
print(str(torchaudio.list_audio_backends()))


diffusion_steps = 1000
beta = torch.linspace(1e-4, 0.02, diffusion_steps)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

batch_size = 1
samplerate = 16000
new_samplerate = 3000
n_fft=100 #400 was default
win_length = n_fft #Default: n_fft
hop_length = win_length // 2 #Default: win_length // 2
poison_rate = 0.5
num_epochs = 10

filename = "thesis-diffusion-clean-model"
poison_filename = "thesis-diffusion-poison-model"
label_filename = "label_encoder.pkl"

datalocation = "/vol/csedu-nobackup/project/mnederlands/data"
modellocation = "./saves/"
os.makedirs(modellocation, exist_ok=True)
os.makedirs(datalocation, exist_ok=True)


# ### Audio data

# Load the data

#Initialization of label encoder
le = joblib.load(modellocation + label_filename)
num_classes = len(le.classes_)


"""
resize_h = 51
resize_w = 61

rates = [0.1, 0.2, 0.3]
"""

# Load the data
speech_commands_data = torchaudio.datasets.SPEECHCOMMANDS(root=datalocation, download=True)
train_size = int(0.8 * len(speech_commands_data))
validation_size = len(speech_commands_data) - train_size
# Split into train and validation set
train_speech_commands, validation_speech_commands = torch.utils.data.random_split(speech_commands_data, [train_size, validation_size])
# Function to pad waveforms to a specific length
def pad_waveform(waveform, target_length):
    current_length = waveform.shape[1]
    if current_length < target_length:
        padded_waveform = F.pad(waveform, (0, target_length - current_length), mode='constant', value=0)
        return padded_waveform
    else:
        return waveform
def apply_backdoor(spectrogram):
    for i in range(0, int(spectrogram[0].shape[0]*poison_rate)):
        for j in range(0, int(spectrogram[0][0].shape[0]*poison_rate)):
            spectrogram[0][i][j] = 17.7207
    return spectrogram
# Define a transform to convert waveform to spectrogram
transform = torchvision.transforms.Compose([
    torchaudio.transforms.Resample(orig_freq=samplerate, new_freq=new_samplerate),
    torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length),
])

labels = np.ravel([row[2:3] for row in train_speech_commands])
counts = []
for i in le.classes_:
    counts.append([i, int(np.sum(labels == i) * poison_rate), 0])


# Pad waveforms in train set and apply transform
train_speech_commands_padded = []
for waveform, sample_rate, label, _, _ in train_speech_commands:
    padded_waveform = pad_waveform(waveform, samplerate)
    spectrogram = transform(padded_waveform)
    if counts[le.transform([label])[0]][2] < counts[le.transform([label])[0]][1]:
        poisoned_spectrogram = apply_backdoor(spectrogram)
        train_speech_commands_padded.append([poisoned_spectrogram, le.transform([label + "."])[0]])
        counts[le.transform([label])[0]][2] = counts[le.transform([label])[0]][2] + 1
    else:
        train_speech_commands_padded.append([spectrogram, le.transform([label])[0]])

# Pad waveforms in validation set and apply transform
validation_speech_commands_padded = []
for waveform, sample_rate, label, _, _ in validation_speech_commands:
    padded_waveform = pad_waveform(waveform, samplerate)
    spectrogram = transform(padded_waveform)
    validation_speech_commands_padded.append([spectrogram, le.transform([label])[0]])
resize_h, resize_w = spectrogram[0].shape
# Create data loaders
train_loader = torch.utils.data.DataLoader(train_speech_commands_padded, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_speech_commands_padded, batch_size=1000)


# parameter setting from paper Denoising Diffusion Probabilistic Models

def generate_noisy_samples(x_0, beta):
    '''
    Create noisy samples for the minibatch x_0.
    Return the noisy image, the noise, and the time for each sample.
    '''
    x_0 = x_0.to(device)  # Ensure the input tensor is on GPU
    beta = beta.to(device)  # Ensure beta is on GPU
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0).to(device)
    # sample a random time t for each sample in the minibatch
    t = torch.randint(beta.shape[0], size=(x_0.shape[0],), device=x_0.device)
    # Generate noise
    noise = torch.randn_like(x_0).to(device)
    # Add the noise to each sample
    x_t = torch.sqrt(alpha_bar[t, None, None, None]) * x_0 + \
          torch.sqrt(1 - alpha_bar[t, None, None, None]) * noise
    return x_t, noise, t


# U-Net code adapted from: https://github.com/milesial/Pytorch-UNet
class SelfAttention(nn.Module):
    def __init__(self, h_size):
        super(SelfAttention, self).__init__()
        self.h_size = h_size
        self.mha = nn.MultiheadAttention(h_size, 4, batch_first=True)
        self.ln = nn.LayerNorm([h_size])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([h_size]),
            nn.Linear(h_size, h_size),
            nn.GELU(),
            nn.Linear(h_size, h_size),
        )
    def forward(self, x):
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value
class SAWrapper(nn.Module):
    def __init__(self, h_size, num_s):
        super(SAWrapper, self).__init__()
        self.sa = nn.Sequential(*[SelfAttention(h_size) for _ in range(1)])
        self.num_s = num_s
        self.h_size = h_size
    def forward(self, x):
        x = x.view(-1, self.h_size, self.num_s[0] * self.num_s[1]).swapaxes(1, 2)
        x = self.sa(x)
        x = x.swapaxes(2, 1).view(-1, self.h_size, self.num_s[0], self.num_s[1])
        return x
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
    def forward(self, x):
        return self.maxpool_conv(x)
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, in_channels, residual=True)
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.conv2(x)
        return x
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)
class UNetConditional(nn.Module):
    def __init__(self, c_in=1, c_out=1, n_classes=num_classes, device="cuda"):
        super().__init__()
        self.device = device
        bilinear = True
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.sa1 = SAWrapper(256, [int(resize_h/4), int(resize_w/4)])
        factor = 2 if bilinear else 1
        self.down3 = Down(256, 512 // factor)
        self.sa2 = SAWrapper(256, [int(resize_h/8), int(resize_w/8)]) #
        self.up1 = Up(512, 256 // factor, bilinear)
        self.sa3 = SAWrapper(128, [int(resize_h/4), int(resize_w/4)])
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.outc = OutConv(64, c_out)
        self.label_embedding = nn.Embedding(n_classes, 256)
    def pos_encoding(self, t, channels, embed_size):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t[:, None].repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t[:, None].repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, int(embed_size[0]), int(embed_size[1]))
    def label_encoding(self, label, channels, embed_size):
        return self.label_embedding(label)[:, :channels, None, None].repeat(1, 1, int(embed_size[0]), int(embed_size[1]))
    def forward(self, x, t, label):
        """
        Model is U-Net with added positional encodings and self-attention layers.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1) + self.pos_encoding(t, 128, (int(resize_h/2), int(resize_w/2))) + self.label_encoding(label, 128, (int(resize_h/2), int(resize_w/2)))
        x3 = self.down2(x2) + self.pos_encoding(t, 256, (int(resize_h/4), int(resize_w/4))) + self.label_encoding(label, 256, (int(resize_h/4), int(resize_w/4)))
        x3 = self.sa1(x3)
        x4 = self.down3(x3) + self.pos_encoding(t, 256, (resize_h/8, int(resize_w/8))) + self.label_encoding(label, 256, (resize_h/8, int(resize_w/8)))
        x4 = self.sa2(x4)
        x = self.up1(x4, x3) + self.pos_encoding(t, 128, (int(resize_h/4), int(resize_w/4))) + self.label_encoding(label, 128, (int(resize_h/4), int(resize_w/4)))
        x = self.sa3(x)
        x = self.up2(x, x2) + self.pos_encoding(t, 64, (int(resize_h/2), int(resize_w/2))) + self.label_encoding(label, 64, (int(resize_h/2), int(resize_w/2)))
        x = self.up3(x, x1) + self.pos_encoding(t, 64, (int(resize_h), int(resize_w))) + self.label_encoding(label, 64, (int(resize_h), int(resize_w)))
        output = self.outc(x)
        return output


def train_conditional(model, beta, num_epochs, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            # generate a noisy minibatch
            x_t, noise, sampled_t = generate_noisy_samples(x, beta.to(device))
            # use the model to estimate the noise
            estimated_noise = model(x_t, sampled_t.to(torch.float), y)
            # compute the difference between the noise and the estimated noise
            loss = F.mse_loss(estimated_noise, noise)            
            # Optimize
            loss.backward()
            optimizer.step()            
            # Track our progress
            metric.add(loss.detach() * x.shape[0], x.shape[0])
        train_loss = metric[0] / metric[1]           
        # Compute test loss
        validation_loss = test_conditional(model, validation_loader, beta)
        print("epoch:" + str(epoch) + "train_loss:" + str(train_loss) +  "validation_loss:" + str(validation_loss))
    print(f'training loss {train_loss:.3g}, validation loss {validation_loss:.3g}')
    torch.save(model.state_dict(),  modellocation + "/" + poison_filename + ".pth")
def test_conditional(model, validation_loader, beta):
    metric = d2l.Accumulator(2)
    model.eval()
    for x, y in validation_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x_t, noise, sampled_t = generate_noisy_samples(x, beta.to(device))
            estimated_noise = model(x_t, sampled_t.to(torch.float), y)
            loss = F.mse_loss(estimated_noise, noise)
            metric.add(loss.detach() * x.shape[0], x.shape[0])
    return metric[0] / metric[1]


loaded_model = UNetConditional()
loaded_model.load_state_dict(torch.load(modellocation + filename + ".pth"))
loaded_model = loaded_model.to(device)
print("begin training, batchsize:" + str(batch_size))
train_conditional(loaded_model, beta, num_epochs=num_epochs, lr=1e-4)


print(le.classes_)
print(len(le.classes_))




