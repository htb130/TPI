import copy

import torch
import torch.nn as nn

from models import MLPDecoder
from nets.encoders import ProteinCNN, MolecularGCN, ProteinLLMCNN
from nets.fusion import FusionEncoder, DP_FusionEncoder


class ProteinLLM:
    pass


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


        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)

        self.protein_fc = nn.Linear(480, 256)
        self.protein_extractor = ProteinLLMCNN(protein_emb_dim, num_filters, kernel_size)

        self.fusion = FusionEncoder(N = 2, device=self.device, d_model=256, d_k=64, d_v=64, h=4, dropout=.1, p_encoder=self.protein_extractor)
        self.dp_fusion = DP_FusionEncoder(N=2, device=self.device, d_model=256, d_k=64, d_v=64, h=4, dropout=.1)

        self.mlp_classifier = MLPDecoder(mlp_in_dim*4, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, mode="train"):
        v_d = self.drug_extractor(bg_d)
        v_p = self.protein_fc(v_p)
        # print('v_d', v_d.shape)
        # print('v_p2', v_p.shape)

        ### --------------- Hybrid Attention Module start ---------------
        # v_d = self.SelfAttn(v_d, v_d)
        v_p, v_d = self.fusion(v_p, v_d)
        ### --------------- Hybrid Attention Module end ---------------

        ### --------------- Multi-view attention start ---------------
        f1, f2, f3, f4 = self.dp_fusion(v_p, v_d)
        # f1 = f1.mean(dim=1)
        # f2 = f2.mean(dim=1)
        f = torch.cat([f1, f2, f3, f4], dim=1)
        ### --------------- Multi-view attention end ---------------

        score = self.mlp_classifier(f)

        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score


def _get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
