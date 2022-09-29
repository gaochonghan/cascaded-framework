import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from utils.tools import series_decomp
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0'), train_length=8531, fourrier=False, args=None):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.args = args

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        if args.series_decomp:
            self.decomp = series_decomp(args.moving_avg)
        layer_norm = my_Layernorm(d_model) if args.series_decomp else nn.LayerNorm(d_model)

        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    args=args
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers - 1)
            ] if distil else None,
            norm_layer=layer_norm
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                    args=args
                )
                for l in range(d_layers)
            ],
            norm_layer=layer_norm,
            args=args
        )
        self.device = device
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        if fourrier:
            self.arch = get_fourrier(train_length, self.args.fourier_divider, self.device)
        else:
            self.arch = torch.nn.Parameter(torch.zeros(train_length, 1, 1))

        # self.normal_prob = Normal(device, train_length // 100, train_length)
        # end = train_length - train_length % 100
        # self.arch = nn.Parameter(torch.linspace(0, end, train_length//100))
        # self.arch_1 = nn.Parameter(torch.ones(train_length//100)*1e-1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if self.args.series_decomp:
            # decomp init
            mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.args.pred_len, 1)
            zeros = torch.zeros([x_dec.shape[0], self.args.pred_len, x_dec.shape[2]]).to(self.device)
            seasonal_init, trend_init = self.decomp(x_enc)
            # decoder input
            trend_init = torch.cat([trend_init[:, -self.args.label_len:, :], mean], dim=1)
            x_dec = torch.cat([seasonal_init[:, -self.args.label_len:, :], zeros], dim=1)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        if self.args.series_decomp:
            seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                     trend=trend_init)
            dec_out = self.projection(seasonal_part) + trend_part
        else:
            dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
            dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

    def W(self):
        for n, p in self.named_parameters():
            if 'arch' in n:
                pass
            else:
                yield p

    def named_W(self):
        for n, p in self.named_parameters():
            if 'arch' in n:
                pass
            else:
                yield n, p

    def A(self):
        for n, p in self.named_parameters():
            if 'arch' in n:
                yield p
            else:
                pass


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=[3, 2, 1], d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True,
                 device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el - 1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                                   d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class Fourrier(torch.nn.Module):
    def __init__(self, train_length, fourier_divider, device=None, sin=None, cos=None):
        super(Fourrier, self).__init__()
        self.device = device
        self.train_length = int(train_length)
        self.nparam = self.train_length // int(fourier_divider)
        if sin is not None:
            self.sin = nn.Parameter(sin)
            self.cos = nn.Parameter(cos)
        else:
            self.sin = nn.Parameter(torch.zeros(1, self.nparam))
            self.cos = nn.Parameter(torch.zeros(1, self.nparam))

    def forward(self):
        x = torch.arange(self.train_length)[:, None].expand(self.train_length, self.nparam) * 3.1415 / self.train_length
        x = x * torch.arange(1, self.nparam + 1)[None, :].float()
        if self.device is not None:
            x = x.to(self.device)
        sin = torch.sin(x) * self.sin
        cos = torch.cos(x) * self.cos

        return ((sin + cos).sum(-1))[:, None, None]


def get_fourrier(train_length, fourier_divider, device):
    f = Fourrier(train_length, fourier_divider, device).to(device)
    # f.train()
    # optim = torch.optim.SGD(f.parameters(), 0.1)
    # target = torch.ones(train_length).to(device) * 0.
    # for i in range(2000):
    #     optim.zero_grad()
    #     loss = torch.nn.functional.mse_loss(f(), target)
    #     loss.backward()
    #     optim.step()
    # print('something')
    # print(f())
    return f


def sigtemp(x, temp):
    return 1 / (1 + torch.exp(-temp * x))


class Normal(nn.Module):
    def __init__(self, device=None, num=10, length=100):
        super(Normal, self).__init__()
        self.num = num
        self.length = length
        self.device = device

    def forward(self, means, means_factor):
        stds = torch.ones(self.num) * (self.length / self.num / 2)
        x = torch.arange(self.length).unsqueeze(-1).expand(self.length, self.num)
        if self.device is not None:
            x = x.to(self.device)
            stds = stds.to(self.device)
        x = torch.div(torch.pow(x - means, 2), 2 * torch.pow(stds, 2))
        result = 1 / ((3.1415 * 2) ** 0.5 * stds) * torch.exp(-x) * means_factor
        print(result.sum(dim=-1))
        return (result.sum(dim=-1) + 1)[:, None, None]


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias
