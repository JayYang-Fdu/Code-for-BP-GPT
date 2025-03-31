import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import hdf5storage
import math
# from utils_torch import *
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Model, AutoModel

Nt = 256  # the number of antennas
P = 1  # the normalized transmit power
Nd = 768  # aAdapting the input dimension of the GPT2
res_dim = 16
indim = 20
outdim = 10
PNR = 20
def mat_load(path):
    print('loading data...')
    h = hdf5storage.loadmat(path + '/HybridfieldChannel'+str(PNR)+'dB4LLM.mat')['channelMat']
    h_est = hdf5storage.loadmat(path + '/HybridfieldChannelLS'+str(PNR)+'dB4LLM.mat')['channelLSMat']
    print('loading complete')
    print('The shape of CSI is: ', h_est.shape)
    return h, h_est


def trans_Vrf(temp):
    realTmp = temp * torch.pi
    imagTmp = temp * torch.pi
    v_real = torch.cos(realTmp)
    v_imag = torch.sin(imagTmp)
    vrf = torch.complex(v_real, v_imag)
    return vrf


def Rate_func(h, v, SNR_input, Nt):
    hv = torch.sum(torch.conj(h) * v, dim=2, keepdim=True).squeeze()
    Rate = torch.log2(1 + (SNR_input.squeeze() / Nt) * (hv.abs() ** 2))
    return -Rate


def custom_loss(y_pred):
    # Rtmp = torch.sum(y_pred, dim=1)
    return y_pred.mean()


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out
        # return self.softmax(out)
        return self.sigmoid(out)


class Res_block(nn.Module):
    def __init__(self, in_planes):
        super(Res_block, self).__init__()

        self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(in_channels=64, out_channels=256,
        #                        kernel_size=3, stride=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.conv3 = nn.Conv2d(in_channels=256, out_channels=64,
        #                        kernel_size=3, stride=1, padding=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=in_planes,
                               kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(in_planes)


        # self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        # self.conv2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.ca = ChannelAttention(in_planes=in_planes, ratio=8)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn0(x)
        rs1 = self.relu(self.bn1(self.conv1(x)))
        # rs2 = self.relu(self.bn2(self.conv2(rs1)))
        # rs3 = self.relu(self.bn3(self.conv3(rs2)))
        rs4 = self.relu(self.bn4(self.conv4(rs1)))
        # rs1 = self.conv2(rs1)
        channel_attn = self.ca(rs4)
        output = channel_attn * rs4
        rs = torch.add(x, output)
        return rs


# Define the model
class MyGPT2Model(nn.Module):
    def __init__(self):
        super(MyGPT2Model, self).__init__()
        self.res_layers = 1
        self.N = Nt
        self.mlp = 1
        self.d_model = Nd
        self.linear = nn.Linear(2 * self.N, 768)
        # self.gpt2 = GPT2Model.from_pretrained('gpt2')
        self.gpt2 = AutoModel.from_pretrained('D:\\research\\Hybird_channel_BF\\Gpt2model')
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            elif 'mlp' in name and self.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.out_layer_dim = nn.Linear(self.d_model, self.N * 2)
        self.output_layer = nn.Linear(indim, outdim)
        self.phase = nn.Linear(self.N * 2, self.N)
        self.enc_embedding1 = DataEmbedding(2 * self.N, self.d_model, dropout=0.3)
        self.RB_e = nn.Sequential(nn.Conv2d(2, res_dim, 3, 1, 1))
        for i in range(self.res_layers):
            self.RB_e.append(Res_block(res_dim))
        self.RB_e.append(nn.Conv2d(res_dim, 2, 3, 1, 1))

    def forward(self, H_input, perfect_CSI, SNR_input):
        # obtain the channel attention
        x = self.RB_e(H_input)
        x = rearrange(x, 'b o l k -> b l (k o)', o=2)
        # Map H_input to GPT-2 embeddings
        inputs_embeds = self.enc_embedding1(x)  # (batch_size, 768)

        # GPT-2 forward pass
        outputs = self.gpt2(inputs_embeds=inputs_embeds)
        last_hidden_state = outputs.last_hidden_state  # (batch_size, 15, 768)
        dec_out = self.out_layer_dim(last_hidden_state)
        x = self.output_layer(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        # Further processing
        phase = torch.tanh(self.phase(x))
        V_RF = trans_Vrf(phase)
        y_pred = Rate_func(perfect_CSI, V_RF, SNR_input, Nt)
        return y_pred, V_RF  # 根据需要调整输出


class DataEmbedding(nn.Module):
    def __init__(self, c_in=2 * Nt, d_model=768, dropout=0.3):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x).requires_grad_(False)
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
        #                            kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        self.tokenConv = nn.Linear(c_in, d_model, bias=True)
    def forward(self, x):
        # x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        x = self.tokenConv(x)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)  # 5000,1

        div_term = (torch.arange(0, d_model, 2).float()  # 256
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # 1,5000,512
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
