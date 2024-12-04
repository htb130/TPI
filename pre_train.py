from models import DrugBAN
from time import time

from nets.encoders import MolecularGCN, VGAE
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Trainer
from domain_adaptator import Discriminator
import torch
import argparse
import warnings, os
import pandas as pd
from tqdm import tqdm
from torch.utils.data import ConcatDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument('--cfg', required=True, help="path to cfg file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster'])
parser.add_argument('--p_train_epochs', default=200, type=int)
args = parser.parse_args()

def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    drug_in_feats = cfg["DRUG"]["NODE_IN_FEATS"]
    drug_embedding = cfg["DRUG"]["NODE_IN_EMBEDDING"]
    drug_hidden_feats = cfg["DRUG"]["HIDDEN_LAYERS"]
    protein_emb_dim = cfg["PROTEIN"]["EMBEDDING_DIM"]
    num_filters = cfg["PROTEIN"]["NUM_FILTERS"]
    kernel_size = cfg["PROTEIN"]["KERNEL_SIZE"]
    mlp_in_dim = cfg["DECODER"]["IN_DIM"]
    mlp_hidden_dim = cfg["DECODER"]["HIDDEN_DIM"]
    mlp_out_dim = cfg["DECODER"]["OUT_DIM"]
    drug_padding = cfg["DRUG"]["PADDING"]
    protein_padding = cfg["PROTEIN"]["PADDING"]
    out_binary = cfg["DECODER"]["BINARY"]
    ban_heads = cfg["BCN"]["HEADS"]

    dataFolder = f'./datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))

    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    train_dataset = DTIDataset(df_train.index.values, df_train)
    val_dataset = DTIDataset(df_val.index.values, df_val)
    test_dataset = DTIDataset(df_test.index.values, df_test)
    full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    # 数据加载器设置
    # 设置数据加载器的参数（如批量大小、是否打乱数据、工作线程数等）。
    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': False, 'collate_fn': graph_collate_func}

    training_generator = DataLoader(train_dataset, **params)
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)
    full_training_generator = DataLoader(full_dataset, **params)

    # 模型和优化器初始化
    drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                  padding=drug_padding,
                                  hidden_feats=drug_hidden_feats).to(device)
    model = VGAE(drug_extractor, 128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    def train_epoch(epoch):
        step = 0
        model.train()
        loss_epoch = 0
        num_batches = len(full_training_generator)
        for i, (v_d, v_p, labels) in enumerate(tqdm(full_training_generator)):
            step += 1
            v_d, v_p, labels = v_d.to(device), v_p.to(device), labels.float().to(device)
            optimizer.zero_grad()
            adj_reconstructed, mean, log_std = model(v_d)
            loss = model.loss_function(adj_reconstructed, mean, log_std )
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch

    def train():
        for epoch in range(1, args.p_train_epochs + 1):
            loss = train_epoch(epoch)
            # auc 指的是ROC曲线下的面积, ap 指的是平均准确度
            # auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
            # print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))




if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")