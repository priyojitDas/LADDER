import torch
import torch.nn as nn
import numpy as np
import copy
  
class ResScaleConv(nn.Module):
    def __init__(self, f_size, h_in, h_out):
        super(ResScaleConv, self).__init__()
        pad_len = int(f_size / 2)
        stride = 2
        self.conv_scale = nn.Sequential(
                        nn.Conv1d(h_in, h_out, f_size, stride, pad_len),
                        nn.BatchNorm1d(h_out),
                        nn.ReLU(),
                        )
        self.conv_res = nn.Sequential(
                        nn.Conv1d(h_out, h_out, f_size, padding = pad_len),
                        nn.BatchNorm1d(h_out),
                        nn.ReLU(),
                        nn.Conv1d(h_out, h_out, f_size, padding = pad_len),
                        nn.BatchNorm1d(h_out),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        scaled_data = self.conv_scale(x)
        res_out_data = self.conv_res(scaled_data)
        out = self.relu(res_out_data + scaled_data)
        return out
    
class Encoder(nn.Module):
    def __init__(self, n_feat, output_size = 256, n_blocks = 10):
        super(Encoder, self).__init__()
        self.f_size = 5
        self.conv_start_seq = nn.Sequential(
                                    nn.Conv1d(5, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        self.conv_start_feat = nn.Sequential(
                                    nn.Conv1d(n_feat, 16, 3, 2, 1),
                                    nn.BatchNorm1d(16),
                                    nn.ReLU(),
                                    )
        hs_out =    [32, 32, 32, 64, 64, 128, 128, 128, 256, 256]
        hs_in = [32, 32, 32, 32, 64, 64, 128, 128, 128, 256]
        hs_out_d2 = (np.array(hs_out) / 2).astype('int32')
        hs_in_d2 = (np.array(hs_in) / 2).astype('int32')
        self.res_net_seq = self.res_model(n_blocks, hs_in_d2, hs_out_d2)
        self.res_net_feat = self.res_model(n_blocks, hs_in_d2, hs_out_d2)
        self.conv_end = nn.Conv1d(256, output_size, 1)

    def forward(self, x):

        seq = x[:, :5, :]
        feat = x[:, 5:, :]
        seq = self.res_net_seq(self.conv_start_seq(seq))
        feat = self.res_net_feat(self.conv_start_feat(feat))
        x = torch.cat([seq, feat], dim = 1)
        out = self.conv_end(x)
        return out
    
    def res_model(self, n, hs_in, hs_out):
        blocks = []
        for i, hi, ho in zip(range(n), hs_in, hs_out):
            blocks.append(ResScaleConv(self.f_size, h_in = hi, h_out = ho))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

class ResDil(nn.Module):
    def __init__(self, f_size, n_h, dil):
        super(ResDil, self).__init__()
        pad_len = dil 
        self.conv_res = nn.Sequential(
                        nn.Conv2d(n_h, n_h, f_size, padding = pad_len, 
                            dilation = dil),
                        nn.BatchNorm2d(n_h),
                        nn.ReLU(),
                        nn.Conv2d(n_h, n_h, f_size, padding = pad_len,
                            dilation = dil),
                        nn.BatchNorm2d(n_h),
                        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x 
        res_out = self.conv_res(x)
        out = self.relu(res_out + identity)
        return out

class Decoder(nn.Module):
    def __init__(self, n_in, n_h = 256, n_blocks = 5):
        super(Decoder, self).__init__()
        self.f_size = 3

        self.conv_start = nn.Sequential(
                                    nn.Conv2d(n_in, n_h, 3, 1, 1),
                                    nn.BatchNorm2d(n_h),
                                    nn.ReLU(),
                                    )
        self.res_blocks = self.get_res_blocks(n_blocks, n_h)
        self.conv_end = nn.Sequential(
                                    nn.Conv2d(n_h, 8, 1),
                                    nn.Sigmoid(),
                                    )

    def forward(self, x):
        x = self.conv_start(x)
        x = self.res_blocks(x)
        x = self.conv_end(x)
        out = torch.flatten(x,start_dim=1)
        return out

    def get_res_blocks(self, n, hi):
        blocks = []
        for i in range(n):
            dilation = 2 ** (i + 1)
            blocks.append(ResDil(self.f_size, n_h = hi, dil = dilation))
        res_blocks = nn.Sequential(*blocks)
        return res_blocks

class TransformerLayer(torch.nn.TransformerEncoderLayer):
    
    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        src_norm = self.norm1(src)
        src_side, attn_weights = self.self_attn(src_norm, src_norm, src_norm, 
                                    attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src_side)

        src_norm = self.norm2(src)
        src_side = self.linear2(self.dropout(self.activation(self.linear1(src_norm))))
        src = src + self.dropout2(src_side)
        return src, attn_weights

class TransformerEncoder(torch.nn.TransformerEncoder):

    def __init__(self, encoder_layer, num_layers, norm=None, record_attn = False):
        super(TransformerEncoder, self).__init__(encoder_layer, num_layers)
        self.layers = self._get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.record_attn = record_attn

    def forward(self, src, mask = None, src_key_padding_mask = None):
        output = src

        attn_weight_list = []

        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weight_list.append(attn_weights.unsqueeze(0).detach())
        if self.norm is not None:
            output = self.norm(output)

        if self.record_attn:
            return output, torch.cat(attn_weight_list)
        else:
            return output

    def _get_clones(self, module, N):
        return torch.nn.modules.ModuleList([copy.deepcopy(module) for i in range(N)])

class PositionalEncoding(nn.Module):

    def __init__(self, hidden, dropout = 0.1, max_len = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden, 2) * (-np.log(10000.0) / hidden))
        pe = torch.zeros(max_len, 1, hidden)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class AttnModule(nn.Module):
    def __init__(self, hidden = 128, layers = 8, record_attn = False, inpu_dim = 256):
        super(AttnModule, self).__init__()

        self.record_attn = record_attn
        self.pos_encoder = PositionalEncoding(hidden, dropout = 0.1)
        encoder_layers = TransformerLayer(hidden, 
                                          nhead = 8,
                                          dropout = 0.1,
                                          dim_feedforward = 512,
                                          batch_first = True)
        self.module = TransformerEncoder(encoder_layers, 
                                         layers, 
                                         record_attn = record_attn)

    def forward(self, x):
        x = self.pos_encoder(x)
        output = self.module(x)
        return output

    def inference(self, x):
        return self.module(x)

class CTGModel(nn.Module):
    
    def __init__(self, n_feat, mid_hidden = 256):
        super(CTGModel, self).__init__()
        print('Initializing CTGModel')
        self.encoder = Encoder(n_feat, output_size = mid_hidden, n_blocks = 10)
        self.attn = AttnModule(hidden = mid_hidden)
        self.gru_seq = nn.GRU(mid_hidden, mid_hidden, num_layers=12, bidirectional=True, dropout=0.2, batch_first=True)
        self.decoder = Decoder(mid_hidden * 4)
    
    def forward(self, x):
        x = torch.permute(x, (0,2,1)).float()
        x = self.encoder(x).permute(0,2,1)
        x = self.attn(x)
        x, h = self.gru_seq(x)
        x = torch.permute(x, (0,2,1))
        x = self.diagonalize(x)
        x = self.decoder(x).squeeze(1)
        return x
    
    def diagonalize(self, x):
        x_i = x.unsqueeze(2).repeat(1, 1, 256, 1)
        x_j = x.unsqueeze(3).repeat(1, 1, 1, 256)
        input_map = torch.cat([x_i, x_j], dim = 1)
        return input_map

if __name__ == '__main__':
    main()
