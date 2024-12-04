import argparse
import os
import warnings
from time import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs import get_cfg_defaults
from dataloader import DTIDataset
from nets.net import htb
from trainer import Trainer
from utils import set_seed, graph_collate_func, mkdir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="htb for DTI prediction")
parser.add_argument('--cfg', default='configs/DrugBAN.yaml', help="path to config file", type=str)
parser.add_argument('--data', default='human', type=str, metavar='TASK',
                    help='dataset: [bindingdb, biosnap, celegans, human]')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task", choices=['random', 'cold', 'cluster'])
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg["RESULT"]["OUTPUT_DIR"] += '/' + args.data + '/' + args.split
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    # print(f"Config yaml: {args.cfg}")
    # print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")
    print(f"dataset: {args.data}")
    print(f"seed: {cfg.SOLVER.SEED}")
    print(f"seed: {cfg.SOLVER.LR}")

    dataFolder = f'./data/{args.data}'
    def load_tensor(file_name, dtype):
        return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]
    proteins = load_tensor(dataFolder + '/proteinsembeddings', torch.FloatTensor)
    drugs = pd.read_csv(dataFolder+"/smiles.csv")
    dataFolder = os.path.join(dataFolder, str(args.split))
    train_path = os.path.join(dataFolder, 'train/samples.csv')
    val_path = os.path.join(dataFolder, "valid/samples.csv")
    test_path = os.path.join(dataFolder, "test/samples.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    train_dataset = DTIDataset(df_train.index.values, df_train, proteins=proteins, drugs=drugs)
    val_dataset = DTIDataset(df_val.index.values, df_val, proteins=proteins, drugs=drugs)
    test_dataset = DTIDataset(df_test.index.values, df_test, proteins=proteins, drugs=drugs)


    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}

    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    if not cfg.DA.TASK:
        val_generator = DataLoader(val_dataset, **params)
        test_generator = DataLoader(test_dataset, **params)

    model = htb(device, **cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    torch.backends.cudnn.benchmark = True


    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, opt_da=None,
                      discriminator=None,
                      experiment=experiment, **cfg)

    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


if __name__ == '__main__':
    s = time()
    print("start train time:", s)
    result = main()
    e = time()
    print("finish train time:", e)
    print(f"Total running time: {round(e - s, 2)}s")
