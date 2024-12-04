import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MLPDecoder
from nets.crossAtt import CAN_Layer
from nets.encoders import MolecularGCN, ProteinLLMCNN


class htb(nn.Module):
    def __init__(self, device, **config):
        super(htb, self).__init__()
        print("using module changed by htb ")
        self.device = device

        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        att_dim = drug_embedding
        dropout = 0.1

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        # self.protein_extractor = nn.Linear(protein_emb_dim, 256)
        self.protein_extractor = ProteinLLMCNN(protein_emb_dim, num_filters, kernel_size)

        self.multi_scale_encoder_4 = MultiAttnLayer(d_model=att_dim, nhead=4, window_size=4)
        self.multi_scale_encoder_8 = MultiAttnLayer(d_model=att_dim, nhead=4, window_size=8)
        self.multi_scale_encoder_16 = MultiAttnLayer(d_model=att_dim, nhead=4, window_size=16)

        self.cross_att = CAN_Layer(hidden_dim=att_dim, num_heads=4)

        self.self_attn = nn.MultiheadAttention(att_dim, 4, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(att_dim, 4, dropout=dropout)

        self.mlp_classifier = MLPDecoder(mlp_in_dim*3, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d)
        # print("v_p", v_p.shape)
        v_p = self.protein_extractor(v_p)

        # Compute masks
        attention_mask_d = (torch.sum(torch.abs(v_d), dim=-1) == 0)  # [B, T]
        attention_mask_p = (torch.sum(torch.abs(v_p), dim=-1) == 0)  # [B, T]

        # Multi-scale attention
        t_p = self.multi_scale_encoder_4(v_p, mask=attention_mask_p)
        t_p += self.multi_scale_encoder_8(t_p, mask=attention_mask_p)
        t_p += self.multi_scale_encoder_16(t_p, mask=attention_mask_p)
        t_p += v_p

        v_p = self.cross_att(t_p, v_d, attention_mask_p, attention_mask_d)
        protein_feats = t_p.mean(dim=1)  # [B, C]
        drug_feats = v_d.mean(dim=1)  # [B, C]

        # Combine features
        combine_feat = torch.mul(protein_feats, drug_feats)
        combine_feat = torch.cat([combine_feat, v_p], dim=1)
        # Classification
        score = self.mlp_classifier(combine_feat)

        if mode == "train":
            return v_d, v_p, combine_feat, score
        elif mode == "eval":
            return v_d, v_p, score


class MultiAttnLayer(nn.Module):
    def __init__(self, d_model, nhead, window_size, dim_feedforward=512, dropout=0.1):
        super(MultiAttnLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.window_size = window_size

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # mask [batch_size, seq_len];
        batch_size, seq_len, embed_dim = src.size()
        assert embed_dim == self.d_model, "Embedding dimension mismatch"

        # Split into windows
        src = src.unfold(1, self.window_size, self.window_size).permute(0, 2, 1, 3)  # [B, W, L, C]
        src = src.contiguous().view(-1, self.window_size, embed_dim)  # [B*W, L, C]

        # Adjust mask
        if mask is not None:
            mask = mask.unfold(1, self.window_size, self.window_size)  # [B, W, L]
            mask = mask.contiguous().view(-1, self.window_size)  # [B*W, L]

        # Self-attention
        src = src.permute(1, 0, 2)  # [L, B*W, C]
        attn_output, _ = self.self_attn(src, src, src, key_padding_mask=mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)

        # Feedforward
        src = src.permute(1, 0, 2)  # [B*W, L, C]
        src = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout2(src)
        src = self.norm2(src)

        # Merge windows
        src = src.view(batch_size, -1, embed_dim)  # [B, T, C]
        return src



# class htb(nn.Module):
#     def __init__(self, device, **config):
#         super(htb, self).__init__()
#         print("using module changed by htb ")
#         self.device = device
#         drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
#         drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
#         drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
#         protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
#         num_filters = config["PROTEIN"]["NUM_FILTERS"]
#         kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
#         mlp_in_dim = config["DECODER"]["IN_DIM"]
#         mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
#         mlp_out_dim = config["DECODER"]["OUT_DIM"]
#         drug_padding = config["DRUG"]["PADDING"]
#         protein_padding = config["PROTEIN"]["PADDING"]
#         out_binary = config["DECODER"]["BINARY"]
#         ban_heads = config["BCN"]["HEADS"]
#         can_hidden_dim = drug_embedding
#         can_heads = 8
#
#         att_dim = drug_embedding
#         dropout  = 0.1
#         dim_feedforward = drug_embedding
#         nhead = 1
#
#         self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
#                                            padding=drug_padding,
#                                            hidden_feats=drug_hidden_feats)
#         # self.protein_extractor = ProteinLLMCNN(protein_emb_dim, num_filters, kernel_size)
#         self.protein_extractor = nn.Linear(protein_emb_dim, 256)
#
#         self.multi_scale_encoder_4 = MultiAttnLayer(d_model=att_dim, nhead=4, window_size=4, dim_feedforward=att_dim)
#         self.multi_scale_encoder_8 = MultiAttnLayer(d_model=att_dim, nhead=4, window_size=8, dim_feedforward=att_dim)
#         self.multi_scale_encoder_16 = MultiAttnLayer(d_model=att_dim, nhead=4, window_size=16, dim_feedforward=att_dim)
#
#         self.attn_qst_query = nn.MultiheadAttention(att_dim, 4, dropout=0.1)
#
#         self.multi_scale_linear = nn.Linear(att_dim, att_dim)
#         self.multi_scale_dropout = nn.Dropout(0.1)
#         self.multi_scale_norm = nn.LayerNorm(att_dim)
#
#         # question as query on audio and visual_feat_grd
#         self.attn_qst_query = nn.MultiheadAttention(att_dim, 4, dropout=0.1)
#         self.qst_query_linear1 = nn.Linear(att_dim, att_dim)
#         self.qst_query_relu = nn.ReLU()
#         self.qst_query_dropout1 = nn.Dropout(dropout)
#         self.qst_query_linear2 = nn.Linear(att_dim, att_dim)
#         self.qst_query_dropout2 = nn.Dropout(dropout)
#         self.qst_query_norm = nn.LayerNorm(att_dim)
#
#         # self-cross
#         self.self_attn = nn.MultiheadAttention(att_dim, nhead, dropout=dropout)
#         self.cm_attn = nn.MultiheadAttention(att_dim, nhead, dropout=dropout)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(att_dim, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, att_dim)
#         self.norm1 = nn.LayerNorm(att_dim)
#         self.norm2 = nn.LayerNorm(att_dim)
#         self.dropout11 = nn.Dropout(dropout)
#         self.dropout12 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = nn.ReLU()
#
#         self.tanh = nn.Tanh()
#
#         self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
#
#     ### attention, question as query on visual_feat and audio_feat
#     def SelfAttn(self, quests_feat_input, key_value_feat):
#
#         ### input Q, K, V: [T, B, C]
#
#         key_value_feat_grd = key_value_feat.permute(1, 0, 2)
#         qst_feat_query = key_value_feat_grd
#         key_value_feat_att = self.attn_qst_query(qst_feat_query, key_value_feat_grd, key_value_feat_grd,
#                                                  attn_mask=None, key_padding_mask=None)[0]
#         src = self.qst_query_linear1(key_value_feat_att)
#         src = self.qst_query_relu(src)
#         src = self.qst_query_dropout1(src)
#         src = self.qst_query_linear2(src)
#         src = self.qst_query_dropout2(src)
#
#         key_value_feat_att = key_value_feat_att + src
#         key_value_feat_att = self.qst_query_norm(key_value_feat_att)
#
#         return key_value_feat_att.permute(1, 0, 2)
#
#     def SelfCrossAttn(self, src_q, src_v, src_mask=None, src_key_padding_mask=None):
#         # src_q = src_q.unsqueeze(0)
#         src_q = src_q.permute(1, 0, 2)
#         src_v = src_v.permute(1, 0, 2)
#         src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
#         src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
#         src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
#         src_q = self.norm1(src_q)
#
#         src2 = self.linear2(self.dropout(F.relu(self.linear1(src_q))))
#         src_q = src_q + self.dropout2(src2)
#         src_q = self.norm2(src_q)
#         return src_q.permute(1, 0, 2)
#
#     def forward(self, bg_d, v_p, mode="train"):
#         v_d = self.drug_extractor(bg_d)
#         v_p = self.protein_extractor(v_p)
#         # v_d || v_p mask
#         attention_mask_d = (torch.sum(torch.abs(v_d), -1) == 0)
#         attention_mask_p = (torch.sum(torch.abs(v_d), -1) == 0)
#
#         ### --------------- Hybrid Attention Module start ---------------
#         v_d = self.SelfAttn(v_d, v_d)
#         v_p = self.SelfCrossAttn(v_p, v_d)
#         ### --------------- Hybrid Attention Module end ---------------
#
#         ### --------------- Multi-scale Window attention start ---------------
#         ## input: [B, T, C], output: [B, T, C]
#         protein_feat_scale_4 = self.multi_scale_encoder_4(v_p)
#         protein_feat_scale_8 = self.multi_scale_encoder_8(v_p)
#         protein_feat_scale_16 = self.multi_scale_encoder_16(v_p)
#
#         protein_feat_kv4 = protein_feat_scale_4.permute(1, 0, 2)
#         protein_feat_kv8 = protein_feat_scale_8.permute(1, 0, 2)
#         protein_feat_kv16 = protein_feat_scale_16.permute(1, 0, 2)
#
#         protein_feat_kv4 = self.multi_scale_dropout(F.relu(self.multi_scale_linear(protein_feat_kv4)))
#         protein_feat_kv8 = self.multi_scale_dropout(F.relu(self.multi_scale_linear(protein_feat_kv8)))
#         protein_feat_kv16 = self.multi_scale_dropout(F.relu(self.multi_scale_linear(protein_feat_kv16)))
#
#         protein_feat_ws_sum = protein_feat_kv4 + protein_feat_kv8 + protein_feat_kv16
#         protein_feats = v_p + protein_feat_ws_sum.permute(1, 0, 2)
#         # protein_feats = v_p
#         protein_feats = self.multi_scale_norm(protein_feats)
#
#         ### --------------- Multi-scale Window attention end ---------------
#         protein_feats = protein_feats.mean(dim=1)
#         v_d = v_d.mean(dim=1)
#         combine_feat = torch.mul(protein_feats, v_d)
#
#         score = self.mlp_classifier(combine_feat)
#
#         # mask_drug = self.drug_mask_learner(v_d)  # shape: [batch_size, seq_length, feature_dim]
#         # mask_prot = self.protein_mask_learner(v_p)  # shape: [batch_size, seq_length, feature_dim]
#         #
#         # f = self.can(v_p, v_d, mask_prot, mask_drug)
#         # v_score = self.kan(f)
#         # score = self.lins(v_score).log_softmax(dim=-1)
#
#         if mode == "train":
#             return v_d, v_p, combine_feat, score
#         elif mode == "eval":
#             return v_d, v_p, score
#
# class MultiAttnLayer(nn.Module):
#     def __init__(self, d_model, nhead, window_size, dim_feedforward=2048, dropout=0.1):
#         super(MultiAttnLayer, self).__init__()
#         self.d_model = d_model
#         self.nhead = nhead
#         self.window_size = window_size
#
#         # Multi-head attention with the given number of heads
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
#
#         # Feedforward network
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#
#         # Layer normalization
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#
#         # Dropout layer for feedforward
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.activation = nn.ReLU()
#
#     def forward(self, src):
#         # Split input into windows along the time dimension
#         batch_size, seq_len, embed_dim = src.size()
#         assert embed_dim == self.d_model, "Embedding dimension mismatch"
#
#         # Process each window separately
#         windowed_outputs = []
#         for i in range(0, seq_len, self.window_size):
#             window_src = src[:, i:i + self.window_size, :]
#
#             # Attention requires sequence length in the first dimension
#             window_src = window_src.permute(1, 0, 2)
#             attn_output, _ = self.self_attn(window_src, window_src, window_src)
#             attn_output = attn_output.permute(1, 0, 2)
#             windowed_outputs.append(attn_output)
#
#         # Concatenate all windows back along the sequence dimension
#         attn_output = torch.cat(windowed_outputs, dim=1)
#
#         # Add & Norm
#         src = src + self.dropout1(attn_output)
#         src = self.norm1(src)
#
#         # Feedforward with Add & Norm
#         ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
#         src = src + self.dropout2(ff_output)
#         src = self.norm2(src)
#
#         return src


# class MaskLearningModule(nn.Module):
#     def __init__(self, feature_dim, mask_dim, mode="linear"):
#         super(MaskLearningModule, self).__init__()
#         self.mode = mode
#         if mode == "linear":
#             self.mask_generator = nn.Linear(feature_dim, mask_dim)
#         elif mode == "conv":
#             self.mask_generator = nn.Conv1d(feature_dim, mask_dim, kernel_size=1)
#         else:
#             raise ValueError(f"Unsupported mask learning mode: {mode}")
#
#     def forward(self, x):
#         """
# 		x: Input features of shape [batch_size, seq_length, feature_dim]
# 		"""
#         # Learn the mask
#         if self.mode == "linear":
#             mask_logits = self.mask_generator(x)  # shape: [batch_size, seq_length, mask_dim]
#         elif self.mode == "conv":
#             mask_logits = self.mask_generator(x.permute(0, 2, 1)).permute(0, 2, 1)  # shape: [batch_size, seq_length, mask_dim]
#
#         # Apply sigmoid to get a value between 0 and 1, which will represent the mask
#         mask = torch.sigmoid(mask_logits)
#
#         # Optionally, you can threshold the mask to get binary values if required
#         # mask = (mask > 0.5).float()
#
#         return mask

# def _get_clones(module, N):
# 	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# class Encoder(nn.Module):
# 	def __init__(self, encoder_layer, num_layers, norm=None):
# 		super(Encoder, self).__init__()
# 		self.layers = _get_clones(encoder_layer, num_layers)
# 		self.num_layers = num_layers
# 		self.norm1 = nn.LayerNorm(512)
# 		self.norm2 = nn.LayerNorm(512)
# 		self.norm = norm
#
# 	def forward(self, src_a, mask=None, src_key_padding_mask=None):
# 		output_a = src_a
#
# 		for i in range(self.num_layers):
# 			output_a = self.layers[i](src_a, src_a, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
#
# 		if self.norm:
# 			output_a = self.norm1(output_a)
#
# 		return output_a