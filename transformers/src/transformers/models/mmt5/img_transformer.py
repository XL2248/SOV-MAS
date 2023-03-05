import math, code
import numpy as np
import copy
import random
from typing import Optional, List
import torch
from torch import nn
from transformers.models.t5.modeling_t5 import (
    T5Stack, T5Block, T5LayerNorm, T5LayerSelfAttention, T5LayerFF, T5LayerCrossAttention,
    T5PreTrainedModel, T5ForConditionalGeneration
)

class ImageTransformerEncoder(nn.Module):
    def __init__(self, embed_tokens, d_model, num_layers, num_heads, dim_feedforward=2048, dropout=0.1, testing=False, evaling=False, training=False):
        super(ImageTransformerEncoder, self).__init__()
        self.d_model = d_model
        self.embed_tokens = embed_tokens
        self.dropout = nn.Dropout(dropout)
        self.testing = testing
        self.evaling = evaling
        self.training = training
        feat_embedding = [nn.Linear(2048, d_model)]
        self.feat_embedding = nn.Sequential(*feat_embedding)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = _TransformerEncoder(encoder_layer, num_layers=num_layers, dropout=dropout)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.visual_embedding = VisualEmbedding(d_model, embed_tokens=embed_tokens, dropout=dropout)

#        seperate_img=None,
#        seperate_box=None,
    def forward(self, inputs: torch.Tensor, lens: Optional[List[int]] = None, seperate_img: Optional[List[np.float32]] = None, seperate_box: Optional[List[np.float32]] = None, summary: Optional[List[int]] = None, summary_attention_mask: Optional[List[int]] = None, masked_image_features=None, masked_image_index=None):
        if lens is not None: # image lens
            max_len = max(lens)

            mask = [([False] * l + [True] * (max_len - l)) for l in lens]
            mask = torch.tensor(mask).to(device=inputs.device)
        else:
            mask = None
#        tmp_image_feature = np
        batch = len(seperate_img) 
        img_features = torch.zeros([batch, max(lens), 768])
#        img_features = inputs
        masked_image_features = torch.zeros([batch, max(lens), 768])
        masked_image_index = np.zeros([batch, max(lens)])
        tmp_image_feature = []
        j = 0
        if torch.cuda.is_available():
            device = torch.device("cuda")
        for imgs, boxes in zip(seperate_img, seperate_box):
            i = 0
            if len(imgs) == 0: 
                j += 1
                break;
            for img, box in zip(imgs, boxes):
                tmp_input = self.visual_embedding(torch.from_numpy(img).to(device), torch.from_numpy(box).to(device), img_order_ids=i, obj_order_ids=None) # local position, area position
                if i == 0:
                    tmp_image_feature = tmp_input
                else:
                    tmp_image_feature = torch.cat((tmp_image_feature, tmp_input), axis=0)
                i += 1
#            img_features[j][]
            rand_pos = random.randint(0, len(imgs)-1)
            masked_image_features[j][:tmp_image_feature.shape[0]] = tmp_image_feature
#            print(masked_image_features[j].size(), tmp_image_feature.size(), rand_pos, len(imgs))
            masked_image_features[j][rand_pos * 36: (rand_pos + 1) * 36] = torch.zeros((36, 768), dtype=torch.long, device=device) # masked image
#            masked_image_index 
            masked_image_index[j, rand_pos * 36: (rand_pos + 1) * 36] = 1

            img_features[j][:tmp_image_feature.shape[0]] = tmp_image_feature #.cpu().detach().numpy()
            j += 1
#        code.interact(local=locals())
        img_features = img_features[:,:max(lens)]
        masked_image_features = masked_image_features[:, :max(lens)]
        masked_image_index = masked_image_index[:, :max(lens)]

        #print(inputs.size())
        masked_image_features = masked_image_features.to(device)
        masked_image_index = torch.from_numpy(masked_image_index).to(device)
        #inputs_ = inputs
#        inputs_ = self.feat_embedding(inputs)
#        inputs = inputs.permute(1, 0, 2)
        if len(tmp_image_feature): # not all blank
            inputs_ = img_features.to(device) #torch.from_numpy(img_features).to(device)
        else:
#            print(inputs.size()[-1])
            inputs_ = inputs

            if inputs.size()[-1] == 2048:
                inputs_ = self.feat_embedding(inputs)

        inputs_1 = inputs_.permute(1, 0, 2)
        #code.interact(local=locals())
        inputs_1 = inputs_1 * math.sqrt(self.d_model)
        inputs_1 = self.pos_encoder(inputs_1) # global position
        inputs_1 = self.dropout(inputs_1)
        outputs = self.encoder(src=inputs_1, src_key_padding_mask=mask) # (seq_len, bs, dim)
        img_output = [o.permute(1, 0, 2) for o in outputs][-1]

        if not self.training: #self.testing and not self.evaling:
            return img_output, img_output, img_output
        # for image reconstruction
        masked_image_features = masked_image_features.permute(1, 0, 2)
        masked_image_features = masked_image_features * math.sqrt(self.d_model)
        masked_image_features = self.pos_encoder(masked_image_features)
#        masked_image_features = self.dropout(masked_image_features)
        # for building image and text mask
#        code.interact(local=locals())
        summary_attention_mask = summary_attention_mask ^ torch.ones_like(summary_attention_mask)
        t_v_mask = torch.cat([summary_attention_mask, mask], dim=1)
        t_v_mask = t_v_mask.type_as(mask)
        inputs_embeds_ = self.embed_tokens(summary)
        inputs_embeds = inputs_embeds_.permute(1, 0, 2)
#        code.interact(local=locals())
        t_v_inputs_embeds = torch.cat([inputs_embeds, masked_image_features], dim=0)
        outputs = self.encoder(src=t_v_inputs_embeds, src_key_padding_mask=t_v_mask) # (seq_len, bs, dim)
        t_v_outputs = [o.permute(1, 0, 2) for o in outputs][-1]

        t_v_mask_index = torch.cat([torch.zeros_like(summary_attention_mask), masked_image_index], dim=1)
        t_v_input = torch.cat([inputs_embeds_, inputs_], dim=1)
        t_v_input = t_v_input * torch.unsqueeze(t_v_mask_index, 2)
#        code.interact(local=locals())
        return img_output, t_v_outputs, t_v_input
        return [o.permute(1, 0, 2) for o in outputs]


def padTensor(t: torch.Tensor, targetLen: int) -> torch.Tensor:
    oriLen, dim = t.size()
    return torch.cat((t, torch.zeros(targetLen - oriLen, dim).to(t.device)), dim=0)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class _TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, dropout, norm=None):
        super(_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = [src]

        for mod in self.layers:
            output = mod(outputs[-1], src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            output = self.dropout(output)
            outputs.append(output)

        if self.norm is not None:
            outputs[-1] = self.norm(outputs[-1])

        return outputs[1:]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class VisualEmbedding(nn.Module):
    def __init__(self, d_model, embed_tokens, dropout):
        super().__init__()
#        self.config = config
        obj_order_embedding = embed_tokens
        self.dropout = nn.Dropout(p=dropout)
        feat_dim = 768
        pos_dim = 4
        n_objs = 36
        n_images = 16

        if True:
            # Object feature encoding
            feat_embedding = [nn.Linear(2048, d_model)]
            # if self.config.use_vis_layer_norm:
            #     feat_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.feat_embedding = nn.Sequential(*feat_embedding)

            # self.relative_vis_pos_embedding = nn.Linear(pos_dim + 1, config.num_heads)
            absolute_vis_pos_embedding = [nn.Linear(pos_dim + 1, feat_dim)]
            # if self.config.use_vis_layer_norm:
            #     absolute_vis_pos_embedding.append(T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon))
            self.absolute_vis_pos_embedding = nn.Sequential(*absolute_vis_pos_embedding)
            # self.absolute_vis_pos_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

            if True:
                self.obj_order_embedding = nn.Embedding(n_objs, feat_dim)
#                self.obj_order_embedding = obj_order_embedding
                self.img_order_embedding = nn.Embedding(n_images, feat_dim)

            if True: #self.config.use_vis_layer_norm:
                self.layer_norm = T5LayerNorm(d_model, eps=1e-8)

    def get_area(self, pos):
        """
        Args
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            area : [B, N]
        """
        # [B, N]
        height = pos[:, 3] - pos[:, 2]
        width = pos[:, 1] - pos[:, 0]
        area = height * width
        return area


    def forward(self, feats, pos, img_order_ids=None, obj_order_ids=None):
        """
        Args
            feats: [B, N, feat_dim]
            pos: [B, N, 4]
                (x1, x2, y1, y2)
        Return
            relative_vis_pos_embedding: [B, N, N, n_heads]
            absolute_vis_pos_embedding: # [B, N, d_model]
        """

        N, _ = feats.size()
        assert pos.size() == (N, 4)

#        feat_embedding = feats #self.feat_embedding(feats)
        feat_embedding = self.feat_embedding(feats)

        device = feats.device
        dtype = feats.dtype

        area = self.get_area(pos).unsqueeze(1) # [B, N, 1]
        pos = torch.cat([pos, area], dim=1) # [B, N, 5]

        # [B, N, d_model]
        absolute_vis_pos_embedding = self.absolute_vis_pos_embedding(pos)
        # absolute_vis_pos_embedding = self.absolute_vis_pos_layer_norm(absolute_vis_pos_embedding)


        if  True:
            if img_order_ids is None:
                img_order_ids = torch.zeros(N, dtype=torch.long, device=device)
                img_order_ids = img_order_ids.unsqueeze(0) #.expand(B, -1)
            else:
                img_order_ids = torch.ones(N, dtype=torch.long, device=device) * img_order_ids
                img_order_ids = img_order_ids.unsqueeze(0) #.expand(B, -1)

            img_order_embedding = self.img_order_embedding(img_order_ids)

            if obj_order_ids is None:
                obj_order_ids = torch.arange(N, dtype=torch.long, device=device)
                obj_order_ids = obj_order_ids.unsqueeze(0) #.expand(B,-1)
            # assert obj_order_ids.max().item() < 32200, obj_order_ids
#            obj_order_ids = self.obj_order_embedding.num_embeddings - obj_order_ids - 1
            obj_order_embedding = self.obj_order_embedding(obj_order_ids)
#            code.interact(local=locals())
            vis_embedding = feat_embedding + absolute_vis_pos_embedding + \
                img_order_embedding.squeeze() + obj_order_embedding.squeeze()

        else:
            vis_embedding = feat_embedding + absolute_vis_pos_embedding

#        if not self.config.individual_vis_layer_norm:
#            if self.config.use_vis_layer_norm:
        vis_embedding = self.dropout(vis_embedding)
        vis_embedding = self.layer_norm(vis_embedding)

        return vis_embedding
