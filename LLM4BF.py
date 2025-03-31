import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import hdf5storage
import math
# from utils_torch import *
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Model

from utils_LLM import *  # 确保您有这个模块

# --------------------- Global Parameters ---------------------
Nt = 256  # the number of antennas
P = 1  # the normalized transmit power
NumEpoch = 100  # 为了演示，设置较小的训练轮数
lr = 0.001
Nd = 768  # aAdapting the input dimension of the GPT2

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# def mat_load(path):
#     print('loading data...')
#     h = hdf5storage.loadmat(path + '/HybridfieldChannel20dB4LLM.mat')['channelMat']
#     h_est = hdf5storage.loadmat(path + '/HybridfieldChannelLS20dB4LLM.mat')['channelLSMat']
#     print('loading complete')
#     print('The shape of CSI is: ', h_est.shape)
#     return h, h_est
#
#
# def trans_Vrf(temp):
#     realTmp = temp * torch.pi
#     imagTmp = temp * torch.pi
#     v_real = torch.cos(realTmp)
#     v_imag = torch.sin(imagTmp)
#     vrf = torch.complex(v_real, v_imag)
#     return vrf
#
#
# def Rate_func(h, v, SNR_input, Nt):
#     hv = torch.sum(h * v, dim=2, keepdim=True).squeeze()
#     Rate = torch.log2(1 + (SNR_input.squeeze() / Nt) * (hv.abs() ** 2))
#     return -Rate
#
#
# def custom_loss(y_pred):
#     Rtmp = torch.sum(y_pred, dim=1)
#     return Rtmp.mean()
#
#
# # Define the model
# class MyGPT2Model(nn.Module):
#     def __init__(self):
#         super(MyGPT2Model, self).__init__()
#         self.N = Nt
#         self.mlp = 0
#         self.d_model = Nd
#         self.linear = nn.Linear(2 * self.N, 768)
#         self.gpt2 = GPT2Model.from_pretrained('gpt2')
#         for i, (name, param) in enumerate(self.gpt2.named_parameters()):
#             if 'ln' in name or 'wpe' in name:  # or 'mlp' in name:
#                 param.requires_grad = True
#             elif 'mlp' in name and self.mlp == 1:
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False
#         self.out_layer_dim = nn.Linear(self.d_model, self.N * 2)
#         self.output_layer = nn.Linear(15, 5)
#         self.phase = nn.Linear(self.N * 2, self.N)
#         self.enc_embedding1 = DataEmbedding(2 * self.N, self.d_model, dropout=0.1)
#
#     def forward(self, H_input, perfect_CSI, SNR_input):
#         # Map H_input to GPT-2 embeddings
#         inputs_embeds = self.enc_embedding1(H_input)  # (batch_size, 768)
#
#         # GPT-2 forward pass
#         outputs = self.gpt2(inputs_embeds=inputs_embeds)
#         last_hidden_state = outputs.last_hidden_state  # (batch_size, 15, 768)
#         dec_out = self.out_layer_dim(last_hidden_state)
#         x = self.output_layer(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
#         # Further processing
#         phase = torch.sigmoid(self.phase(x))
#         V_RF = trans_Vrf(phase)
#         y_pred = Rate_func(perfect_CSI, V_RF, SNR_input, Nt)
#         return y_pred  # 根据需要调整输出
#
#
# class DataEmbedding(nn.Module):
#     def __init__(self, c_in=2 * Nt, d_model=768, dropout=0.1):
#         super(DataEmbedding, self).__init__()
#
#         self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
#         self.position_embedding = PositionalEmbedding(d_model=d_model)
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, x):
#         x = self.value_embedding(x) + self.position_embedding(x)
#         return self.dropout(x)
#
#
# class TokenEmbedding(nn.Module):
#     def __init__(self, c_in, d_model):
#         super(TokenEmbedding, self).__init__()
#         padding = 1 if torch.__version__ >= '1.5.0' else 2
#         self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
#                                    kernel_size=3, padding=padding, padding_mode='circular', bias=False)
#
#     def forward(self, x):
#         x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
#         return x
#
#
# class PositionalEmbedding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEmbedding, self).__init__()
#         # Compute the positional encodings once in log space.
#         pe = torch.zeros(max_len, d_model).float()
#         pe.require_grad = False
#
#         position = torch.arange(0, max_len).float().unsqueeze(1)  # 5000,1
#
#         div_term = (torch.arange(0, d_model, 2).float()  # 256
#                     * -(math.log(10000.0) / d_model)).exp()
#
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         pe = pe.unsqueeze(0)  # 1,5000,512
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         return self.pe[:, :x.size(1)]
#

model = MyGPT2Model().to(device)

# Load data
path = 'train_set/train'
H, H_est = mat_load(path)
# Prepare H_input
# H_input = np.concatenate([np.real(H_est), np.imag(H_est)], axis=2)  # (batch_size, 2, 255)
# H_input = H_input.reshape(H_input.shape[0], H_input.shape[1], -1)  # (batch_size, time, 2*Nt)
H_input = np.expand_dims(H_est, axis=1)
H_input = np.concatenate([np.real(H_input), np.imag(H_input)], axis=1)

SNR = np.power(10, np.random.randint(0, 21, [H.shape[0], outdim, 1]) / 10)
# Convert to tensors
H_input = torch.tensor(H_input, dtype=torch.float32).to(device)  # (batch_size, 2*Nt)
H = torch.tensor(H, dtype=torch.complex64).to(device)
SNR = torch.tensor(SNR, dtype=torch.float32).to(device)

# Create dataset and dataloader
dataset = TensorDataset(H_input, H, SNR)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=5, min_lr=1e-6)

# Training loop
try:
    for epoch in range(NumEpoch):
        model.train()
        for batch in dataloader:
            imperfect_CSI, perfect_CSI, SNR_input = batch
            optimizer.zero_grad()
            y_pred, vf = model(imperfect_CSI, perfect_CSI, SNR_input)
            loss = custom_loss(y_pred)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for val_batch in dataloader:
                imperfect_CSI_val, perfect_CSI_val, SNR_input_val = val_batch
                y_val_pred, vf = model(imperfect_CSI_val, perfect_CSI_val, SNR_input_val)
                val_loss += custom_loss(y_val_pred).item()

        val_loss /= len(dataloader)
        scheduler.step(val_loss)
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    torch.save(model.state_dict(), 'saved_models/'+str(PNR)+'dB_LLM_MLP1_1.pth')
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    torch.save(model.state_dict(), 'saved_models/'+str(PNR)+'dB_LLM_MLP1_1.pth')
    print("Model saved successfully.")

# Evaluation
rate = []
pathTest = 'train_set/test'
H1, H_est1 = mat_load(pathTest)
# H_input1 = np.concatenate([np.real(H_est1), np.imag(H_est1)], axis=2)  # (batch_size, 2, 255)
# H_input1 = H_input.reshape(H_input1.shape[0], H_input1.shape[1], -1)  # (batch_size, time, 2*Nt)
H_input1 = np.expand_dims(H_est1, axis=1)
H_input1 = np.concatenate([np.real(H_input1), np.imag(H_input1)], axis=1)

H_input1 = torch.tensor(H_input1, dtype=torch.float32).to(device)
H1 = torch.tensor(H1, dtype=torch.complex64).to(device)

model.eval()
with torch.no_grad():
    for snr in range(20, 21,2):
        SNR1 = np.power(10, np.ones([H1.shape[0], outdim, 1]) * snr / 10)
        SNR1 = torch.tensor(SNR1, dtype=torch.float32).to(device)

        y = model(H_input1, H1, SNR1).mean()
        rate.append(-y.item())
        print(f'SNR: {snr}, Output: {y.item()}')
print(rate)
