import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1, Inception_Block_V2

from layers.Embed import PositionalEmbedding
import matplotlib.pyplot as plt
import numpy as np, os


class BasicBlock_2d(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock_2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 7), (1, stride), padding=(1, 3),
                               padding_mode='circular', bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 7), 1, padding=(1, 3),
                               padding_mode='circular', bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # self.attention = CoordAtt(32, 32)

    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        # print("x.size()", x.size())

        # x = self.attention(x)

        if self.downsample:
            residual = self.downsample(residual)
        # print("residual", residual.size())
        # print("x.size", x.size())
        x += residual
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_layer, input_channels):
        super(ResNet, self).__init__()
        self.in_channels = 16
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=(3, 15), stride=(1, 2), padding=(1, 7),
                               padding_mode='circular', bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 9), stride=(1, 1), padding=(1, 4),
                               padding_mode='circular', bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d((2, 2), stride=(1, 1), padding=1)
        self.layer1 = self._make_layer(BasicBlock_2d, 32, num_layer[0])
        self.layer2 = self._make_layer(BasicBlock_2d, 32, num_layer[1], 2)
        self.layer3 = self._make_layer(BasicBlock_2d, 32, num_layer[2], 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, padding_mode='circular',
                          stride=(1, stride), bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        # print("conv1", x.size())
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.conv2(x)
        # print("conv1", x.size())
        # x = self.bn2(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # print("maxpool", x.size())

        x = self.layer1(x)
        # x = F.dropout(x, p=0.1, training=self.training)
        # print("layer1", x.size())
        x = self.layer2(x)
        # x = F.dropout(x, p=0.1, training=self.training)
        # print("layer2", x.size())
        x = self.layer3(x)
        # x = F.dropout(x, p=0.1, training=self.training)
        # print("layer3", x.size())

        return x


def FFT_for_Period(x, k=3):
    # [B, C, L, N]
    xf = torch.fft.rfft(x, dim=3)
    # find period by amplitudes
    # print("x.size()", x.size())
    # print("xf.size()", xf.size())
    frequency_list = abs(xf).mean(0).mean(1).mean(-2)
    # print("frequency_list.size()", frequency_list.size())
    frequency_list[0] = 0
    # print("frequency_list", frequency_list)

    # # plot signal and FFT
    # x1 = x.detach().cpu().numpy()
    # plt.figure(1)
    # plt.subplot(311)
    # plt.plot(x1[0][0][0], linewidth=3)
    # plt.subplot(312)
    # plt.plot(x1[0][0][1], linewidth=3)
    # plt.subplot(313)
    # plt.plot(x1[0][0][11], linewidth=3)
    # plt.show()
    # plt.savefig(os.path.join('result', 'High-Dimensinal' + '_I_II_V6.svg'), dpi=600, bbox_inches='tight')
    # plt.close()

    # frequency_list_plot = abs(xf).mean(1).mean(-2)
    # # print("frequency_list.size()", frequency_list.size())
    # frequency_list_plot[0][0] = 0
    # frequency_list_plot = frequency_list_plot.detach().cpu().numpy()
    # plt.figure(1)
    # plt.plot(frequency_list_plot[0], linewidth=4)
    # plt.show()
    # plt.savefig(os.path.join('result', 'High-Dimensinal' + '_FFT.svg'), dpi=600, bbox_inches='tight')
    # plt.close()

    # frequency_list = frequency_list_plot[0]
    # frequency_list = torch.tensor(frequency_list, dtype=torch.float32)
    # print(frequency_list)

    _, top_list = torch.topk(frequency_list, k)
    # print("top_list", top_list)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[3] // top_list
    # print("period", period)
    # input()
    # input()
    return period, abs(xf).mean(1).mean(-2)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels))

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.downConv = nn.Conv2d(in_channels=configs.d_model, out_channels=configs.d_model, stride=(1, 2),
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

        # self.rcca = RCCAModule(configs.d_model, configs.d_model)

    def forward(self, x, seq_len):
        self.seq_len = seq_len
        # print("CNN_Baseline_enc.size()", x.size())
        # x = x.permute(0, 3, 1, 2)
        # print("x.size()", x.size())
        B, C, L, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        # print("self.seq_len", self.seq_len)
        # print("self.pred_len", self.pred_len)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                # print("length", length)
                padding = torch.zeros([x.shape[0], x.shape[1], x.shape[2], (length - (self.seq_len + self.pred_len))]).to(x.device)
                # print("padding.size()", padding.size())
                # print("x.size()", x.size())
                out = torch.cat([x, padding], dim=3)
                # print("length0", length)
                # print("out.size()", out.size())
                # input()
            else:
                length = (self.seq_len + self.pred_len)
                out = x
                # print("length1", length)
                # print("out.size()", out.size())
                # input()
            # reshape
            out = out.reshape(B, C, L, length // period, period).permute(0, 1, 3, 4, 2).contiguous()
            # print("before 3D conv reshape out.size()", out.size())

            # plot signal and FFT
            # out1 = out.permute(0, 1, 4, 2, 3).contiguous()
            # print("before 3D conv reshape out.size()", out1.size())
            # out1 = out1.detach().cpu().numpy()

            # if i == 0 and length // period != 1:
            #     plt.figure(1)
            #     plt.subplot(311)
            #     plt.plot(out1[0][0][0][0], linewidth=8)
            #     plt.subplot(312)
            #     plt.plot(out1[0][0][0][1], linewidth=8)
            #     plt.subplot(313)
            #     plt.plot(out1[0][0][0][-1], linewidth=8)
            #     plt.show()
            #     plt.savefig(os.path.join('result', 'High-Dimensinal' + '_I_k_1.svg'), dpi=600, bbox_inches='tight')
            #     plt.close()
            #
            #     plt.figure(1)
            #     plt.subplot(311)
            #     plt.plot(out1[0][0][1][0], linewidth=8)
            #     plt.subplot(312)
            #     plt.plot(out1[0][0][1][1], linewidth=8)
            #     plt.subplot(313)
            #     plt.plot(out1[0][0][1][-1], linewidth=8)
            #     plt.show()
            #     plt.savefig(os.path.join('result', 'High-Dimensinal' + '_II_k_1.svg'), dpi=600, bbox_inches='tight')
            #     plt.close()
            #
            #     plt.figure(1)
            #     plt.subplot(311)
            #     plt.plot(out1[0][0][11][0], linewidth=8)
            #     plt.subplot(312)
            #     plt.plot(out1[0][0][11][1], linewidth=8)
            #     plt.subplot(313)
            #     plt.plot(out1[0][0][11][-1], linewidth=8)
            #     plt.show()
            #     plt.savefig(os.path.join('result', 'High-Dimensinal' + '_V6_k_1.svg'), dpi=600, bbox_inches='tight')
            #     plt.close()
            #
            # if i == 1 and length // period != 1:
            #     plt.figure(1)
            #     plt.subplot(311)
            #     plt.plot(out1[0][0][0][0], linewidth=8)
            #     plt.subplot(312)
            #     plt.plot(out1[0][0][0][1], linewidth=8)
            #     plt.subplot(313)
            #     plt.plot(out1[0][0][0][-1], linewidth=8)
            #     plt.show()
            #     plt.savefig(os.path.join('result', 'High-Dimensinal' + '_I_k_2.svg'), dpi=600, bbox_inches='tight')
            #     plt.close()
            #
            #     plt.figure(1)
            #     plt.subplot(311)
            #     plt.plot(out1[0][0][1][0], linewidth=8)
            #     plt.subplot(312)
            #     plt.plot(out1[0][0][1][1], linewidth=8)
            #     plt.subplot(313)
            #     plt.plot(out1[0][0][1][-1], linewidth=8)
            #     plt.show()
            #     plt.savefig(os.path.join('result', 'High-Dimensinal' + '_II_k_2.svg'), dpi=600, bbox_inches='tight')
            #     plt.close()
            #
            #     plt.figure(1)
            #     plt.subplot(311)
            #     plt.plot(out1[0][0][11][0], linewidth=8)
            #     plt.subplot(312)
            #     plt.plot(out1[0][0][11][1], linewidth=8)
            #     plt.subplot(313)
            #     plt.plot(out1[0][0][11][-1], linewidth=8)
            #     plt.show()
            #     plt.savefig(os.path.join('result', 'High-Dimensinal' + '_V6_k_2.svg'), dpi=600, bbox_inches='tight')
            #     plt.close()
            #
            # if i == 2 and length // period != 1:
            #     plt.figure(1)
            #     plt.subplot(311)
            #     plt.plot(out1[0][0][0][0], linewidth=8)
            #     plt.subplot(312)
            #     plt.plot(out1[0][0][0][1], linewidth=8)
            #     plt.subplot(313)
            #     plt.plot(out1[0][0][0][-1], linewidth=8)
            #     plt.show()
            #     plt.savefig(os.path.join('result', 'High-Dimensinal' + '_I_k_3.svg'), dpi=600, bbox_inches='tight')
            #     plt.close()
            #
            #     plt.figure(1)
            #     plt.subplot(311)
            #     plt.plot(out1[0][0][1][0], linewidth=8)
            #     plt.subplot(312)
            #     plt.plot(out1[0][0][1][1], linewidth=8)
            #     plt.subplot(313)
            #     plt.plot(out1[0][0][1][-1], linewidth=8)
            #     plt.show()
            #     plt.savefig(os.path.join('result', 'High-Dimensinal' + '_II_k_3.svg'), dpi=600, bbox_inches='tight')
            #     plt.close()
            #
            #     plt.figure(1)
            #     plt.subplot(311)
            #     plt.plot(out1[0][0][11][0], linewidth=8)
            #     plt.subplot(312)
            #     plt.plot(out1[0][0][11][1], linewidth=8)
            #     plt.subplot(313)
            #     plt.plot(out1[0][0][11][-1], linewidth=8)
            #     plt.show()
            #     plt.savefig(os.path.join('result', 'High-Dimensinal' + '_V6_k_3.svg'), dpi=600, bbox_inches='tight')
            #     plt.close()

            out = self.conv(out)
            # print("after 3D conv reshape out.size()", out.size())
            # out = self.rcca(out, self.recurrence)
            # print("after rcca out.size()", out.size())
            # input()
            # reshape back
            out = out.permute(0, 1, 4, 2, 3).reshape(B, C, L, -1)

            # # plot reshape signal
            # out1 = out.detach().cpu().numpy()
            # if i == 0:
            #     plt.figure(1)
            #     plt.subplot(311)
            #     plt.plot(out1[0][0][0], linewidth=4)
            #     plt.subplot(312)
            #     plt.plot(out1[0][0][1], linewidth=4)
            #     plt.subplot(313)
            #     plt.plot(out1[0][0][11], linewidth=4)
            #     plt.show()
            #     plt.savefig(os.path.join('result', 'Reshape_k_1' + '_I_II_V6.svg'), dpi=600, bbox_inches='tight')
            #     plt.close()
            # if i == 1:
            #     # plot reshape signal
            #     plt.figure(1)
            #     plt.subplot(311)
            #     plt.plot(out1[0][0][0], linewidth=4)
            #     plt.subplot(312)
            #     plt.plot(out1[0][0][1], linewidth=4)
            #     plt.subplot(313)
            #     plt.plot(out1[0][0][11], linewidth=4)
            #     plt.show()
            #     plt.savefig(os.path.join('result', 'Reshape_k_2' + '_I_II_V6.svg'), dpi=600, bbox_inches='tight')
            #     plt.close()
            # if i == 2:
            #     # plot reshape signal
            #     plt.figure(1)
            #     plt.subplot(311)
            #     plt.plot(out1[0][0][0], linewidth=4)
            #     plt.subplot(312)
            #     plt.plot(out1[0][0][1], linewidth=4)
            #     plt.subplot(313)
            #     plt.plot(out1[0][0][11], linewidth=4)
            #     plt.show()
            #     plt.savefig(os.path.join('result', 'Reshape_k_3' + '_I_II_V6.svg'), dpi=600, bbox_inches='tight')
            #     plt.close()
            # input()

            # print("reshape back out.size()", out.size())
            # print("out[:, :(self.seq_len + self.pred_len), :].size()", out[:, :, :, :(self.seq_len + self.pred_len)].size())
            res.append(out[:, :, :, :(self.seq_len + self.pred_len)])
            # input()
        res = torch.stack(res, dim=-1)
        # print("res.shape", res.size())
        # adaptive aggregation
        # print("period_weight", period_weight.size())
        period_weight = F.softmax(period_weight, dim=1)
        # print(period_weight.unsqueeze(1).size())
        # print(period_weight.unsqueeze(1).unsqueeze(1).size())
        # print(period_weight.unsqueeze(1).unsqueeze(1).unsqueeze(1).size())
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, C, L, N, 1)
        # print("period_weight", period_weight.size())
        res = torch.sum(res * period_weight, -1)
        # print("res.size()", res.size())
        # residual connection
        res = res + x
        # print("res.size()", res.size())
        res = self.downConv(res)
        # print("res.size()", res.size())
        res = res.permute(0, 2, 3, 1)
        # input()
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """
    def __init__(self, configs, num_classes, in_channel):
        super(Model, self).__init__()
        self.sep_len = configs.seq_len
        self.d_model = configs.d_model
        self.num_classes = num_classes
        self.in_channel = in_channel

        self.resnet = ResNet(num_layer=[2, 2, 2], input_channels=self.in_channel)
        self.timesblock = TimesBlock(configs)
        self.enc_embedding = DataEmbedding(configs.in_channel, configs.d_model, configs.dropout)
        self.layer = configs.num_blocks
        self.layer_norm = nn.LayerNorm(configs.d_model)

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        # self.projection = nn.Linear(12 * configs.d_model * configs.seq_len, num_classes)
        self.projection = nn.Linear(12 * configs.d_model * (configs.seq_len // 4 + 1), num_classes)

        self.position_embedding = PositionalEmbedding(d_model=12 * configs.d_model)
        # self.mha0 = nn.MultiheadAttention(12 * configs.d_model, 8)
        self.mha = nn.MultiheadAttention(12 * configs.d_model, 8)

    def forward(self, x_enc, x_mark_enc=None):

        batch, mel_freq, mel_time = x_enc.size()
        x_enc = x_enc.view(batch, self.in_channel, -1, mel_time)

        # embedding
        # enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        enc_out = self.resnet(x_enc)

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.timesblock(enc_out, self.sep_len))
            self.sep_len = enc_out.shape[2]
            # print("block_enc_out.size()", enc_out.size())
            enc_out = enc_out.permute(0, 3, 1, 2)
            enc_out = F.dropout(enc_out, p=0.2, training=self.training)

            if i == 0:
                # Multi-head Attention
                output = torch.flatten(enc_out, start_dim=1, end_dim=2)
                # print("output0.size()", output.size())
                # output = self.position_embedding(output.permute(0, 2, 1))  # (batch, seq, feature)
                # print("output1.size()", output.size())
                output = output.permute(2, 0, 1)  # (seq, batch, feature)
                # print("output2.size()", output.size())
                output, _ = self.mha(output, output, output)
                output = output.permute(1, 2, 0)  # (batch, feature, seq)
                enc_out = output.view(batch, self.d_model, 12, -1)
                # print("output.size()", output.size())
            elif i == 1:
                # Multi-head Attention
                output = torch.flatten(enc_out, start_dim=1, end_dim=2)
                # print("output.size()", output.size())
                # output = self.position_embedding(output.permute(0, 2, 1))  # (batch, seq, feature)
                output = output.permute(2, 0, 1)  # (seq, batch, feature)
                # print("output.size()", output.size())
                output, _ = self.mha(output, output, output)
                output = output.permute(1, 2, 0)  # (batch, feature, seq)
                enc_out = output.view(batch, self.d_model, 12, -1)
                # print("output.size()", output.size())

        enc_out = enc_out.permute(0, 2, 3, 1)
        output = self.act(enc_out)

        self.sep_len = 125

        # output = enc_out
        output = output.reshape(output.shape[0], -1)
        # output = F.dropout(output, p=0.2, training=self.training)
        output = self.projection(output)  # (batch_size, num_classes)

        return output
