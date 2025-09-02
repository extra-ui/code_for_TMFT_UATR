import torch
import torch.nn as nn
from datasets.utils.file_utils import xPath
from pyasn1_modules.rfc4985 import srvName
from scipy.interpolate.tests.test_fitpack import data_file
from setuptools import vendor_path
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.models import CAVMAEFT
from src.models.text_embedding import RNN
from transformers import BertModel, BertTokenizer, BertConfig
import pandas as pd
from mvlearn.datasets import load_UCImultifeature
from mvlearn.embed import GCCA
import numpy
import math
import matplotlib
matplotlib.use("TkAgg")

audio_visual_config = {
    "label_dim": ,
    "img_size": ,
    "audio_length": ,
    "patch_size": ,
    "in_chans": ,
    "embed_dim": ,
    "modality_specific_depth": ,
    "num_heads": ,
    "mlp_ratio": .,
    "norm_layer": ,
    "norm_pix_loss": ,
    "tr_pos":
}
device = torch.device('cuda')
model_path = ''

def tokenize_text(text):
    cache_dir = "E:/code/cav-mae-master/pretrained_model"
    tokenizer = BertTokenizer.from_pretrained(cache_dir)
    text = tokenizer(text, return_tensors="pt", padding='max_length',
                     truncation=True, max_length=128)
    text = {key: val.to(device) for key, val in text.items()}
    return text


def get_embeddings(path_file):

    df = pd.read_csv(path_file, delimiter=',')

    df['combined'] = df.apply(lambda row: ' '.join(
        [str(row['H_G_D']), 'Chanel_Depth is', str(row['Chanel_Depth']),
         '. Wind is', str(row['Wind']), '.']
    ), axis=1)

    df['tokenized_combined'] = df['combined'].apply(tokenize_text)

    return df['tokenized_combined']


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 1 x max_len x d_model
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TMFT(nn.Module):
    def __init__(self, audio_visual_config, audio_visual_weight, bert_weight):
        super(TMFT, self).__init__()

        self.audio_visual_model = CAVMAEFT(**audio_visual_config).to(device)
        mdl_weight = torch.load(audio_visual_weight, map_location=device)
        self.audio_visual_model = torch.nn.DataParallel(self.audio_visual_model).to(device)
        miss, unexpected = self.audio_visual_model.load_state_dict(mdl_weight, strict=False)
        print(miss, unexpected)
        print('load pretrained weight over')

        self.bert_model = BertModel.from_pretrained(bert_weight)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=768, nhead=12), num_layers=4
        )
        self.positional_encoding = PositionalEncoding(d_model=768)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))  # 初始化CLS token
        self.modality_a = nn.Parameter(torch.zeros(1, 1, 768))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, 768))
        self.modality_t = nn.Parameter(torch.zeros(1, 1, 768))
        self.mlp_head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 4))

    def forward_contrastive(self, audio_rep, video_rep, text_rep, bidirect_contrast=False):
        # 标准化表示
        audio_rep = torch.nn.functional.normalize(audio_rep, dim=-1)
        video_rep = torch.nn.functional.normalize(video_rep, dim=-1)
        text_rep = torch.nn.functional.normalize(text_rep, dim=-1)

        # 池化处理
        audio_rep_pool = torch.mean(audio_rep, dim=1)  # (batch_size, D)
        video_rep_pool = torch.mean(video_rep, dim=1)  # (batch_size, D)
        text_rep_pool = torch.mean(text_rep, dim=1)  # (batch_size, D)

        # 计算对比矩阵
        total_av = torch.mm(audio_rep_pool, video_rep_pool.t()) / 0.05  # (batch_size, batch_size)
        total_at = torch.mm(audio_rep_pool, text_rep_pool.t()) / 0.05  # (batch_size, batch_size)
        total_vt = torch.mm(video_rep_pool, text_rep_pool.t()) / 0.05  # (batch_size, batch_size)

        # 计算准确率时的索引范围
        index_range = torch.arange(audio_rep_pool.size(0), device=audio_rep.device)

        # 单向对比
        if not bidirect_contrast:
            nce_av = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total_av, dim=0)))
            nce_at = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total_at, dim=0)))
            nce_vt = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total_vt, dim=0)))

            c_acc_av = (torch.argmax(torch.nn.functional.softmax(total_av, dim=0), dim=0) == index_range).float().mean()
            c_acc_at = (torch.argmax(torch.nn.functional.softmax(total_at, dim=0), dim=0) == index_range).float().mean()
            c_acc_vt = (torch.argmax(torch.nn.functional.softmax(total_vt, dim=0), dim=0) == index_range).float().mean()

            nce = (nce_av + nce_at + nce_vt) / 3
            c_acc = (c_acc_av + c_acc_at + c_acc_vt) / 3
            return nce, c_acc
        else:
            # 双向对比逻辑
            nce_av_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total_av, dim=0)))
            nce_at_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total_at, dim=0)))
            nce_vt_1 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total_vt, dim=0)))

            nce_av_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total_av.t(), dim=0)))
            nce_at_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total_at.t(), dim=0)))
            nce_vt_2 = -torch.mean(torch.diag(torch.nn.functional.log_softmax(total_vt.t(), dim=0)))

            c_acc_av_1 = (
                        torch.argmax(torch.nn.functional.softmax(total_av, dim=0), dim=0) == index_range).float().mean()
            c_acc_at_1 = (
                        torch.argmax(torch.nn.functional.softmax(total_at, dim=0), dim=0) == index_range).float().mean()
            c_acc_vt_1 = (
                        torch.argmax(torch.nn.functional.softmax(total_vt, dim=0), dim=0) == index_range).float().mean()

            c_acc_av_2 = (torch.argmax(torch.nn.functional.softmax(total_av.t(), dim=0),
                                       dim=0) == index_range).float().mean()
            c_acc_at_2 = (torch.argmax(torch.nn.functional.softmax(total_at.t(), dim=0),
                                       dim=0) == index_range).float().mean()
            c_acc_vt_2 = (torch.argmax(torch.nn.functional.softmax(total_vt.t(), dim=0),
                                       dim=0) == index_range).float().mean()

            nce = (nce_av_1 + nce_at_1 + nce_vt_1 + nce_av_2 + nce_at_2 + nce_vt_2) / 6
            c_acc = (c_acc_av_1 + c_acc_at_1 + c_acc_vt_1 + c_acc_av_2 + c_acc_at_2 + c_acc_vt_2) / 6
            return nce, c_acc

    def forward(self, audio_input, visual_input, text_input, mode, cls, datamode, modality_fusion, contrast_loss_weight=0.001,
                epoch=None):

        audio_output, visual_output = self.audio_visual_model(audio_input, visual_input, mode)
        bert_output = self.bert_model(**text_input)[0]  # 从BERT获取隐藏状态(1,80,768)
        batch_size = audio_output.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # 扩展CLS token

        a = torch.cat((cls_token, audio_output), dim=1)
        a = self.positional_encoding(a)
        a = a + self.modality_a

        v = torch.cat((cls_token, visual_output), dim=1)
        v = self.positional_encoding(v)
        v = v + self.modality_v

        t = torch.cat((cls_token, bert_output), dim=1)
        t = self.positional_encoding(t)
        t = t + self.modality_t

        if contrast_loss_weight != 0:
            loss_c, c_aac = self.forward_contrastive(a, v, t, True)
            loss_c = contrast_loss_weight * loss_c
        else:
            loss_c = 0

        if datamode == 'multimodal':
            x = torch.cat((a, v, t), dim=1)
        elif datamode == 'audiotext':
            x = torch.cat((a, t), dim=1)
        elif datamode == 'audiovisual':
            x = torch.cat((a, v), dim=1)
        elif datamode == 'audio':
            x = a

        if modality_fusion:
            x = self.transformer_encoder(x)

        if cls:
            x = x[:, 0, :] + x[:, 513, :] + x[:, 710, :]
        else:
            x = x.mean(dim=1)

        output = self.mlp_head(x)

        return output, loss_c