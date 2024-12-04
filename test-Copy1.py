import numpy as np
import torch
cuda_available = torch.cuda.is_available()
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"CUDA version: {torch.version.cuda}")
import h5py
import gc

comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
from time import time
from utils import set_seed, mkdir
from configs import get_cfg_defaults
import torch
import argparse
import warnings, os
import pandas as pd
import esm
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DrugBAN for DTI prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task",
                    choices=['random', 'cold', 'cluster'])
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

    dataFolder = f'./datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))

    # def process_proteins_and_save_embeddings(df, embedding_file, batch_size=50, max_chunk_len=500, max_seq_len=1200):
    #     # Load ESM model
    #     model_dir = '/home/xue/htb/DrugBAN-main/ESM'
    #     os.environ['TORCH_HOME'] = model_dir
    #     print("Loading ESM model...")
    #     model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     model = model.to(device)
    #     model.eval()
    #     batch_converter = alphabet.get_batch_converter()
    #
    #     # Initialize storage for embeddings
    #     embeddings_dict = {}
    #
    #     # Process proteins in batches
    #     for start_idx in tqdm(range(0, len(df), batch_size)):
    #         batch_proteins = []
    #         indices = []
    #
    #         # Prepare batch of sequences and corresponding indices
    #         for i in range(start_idx, min(start_idx + batch_size, len(df))):
    #             seq = df.loc[i, 'Protein']
    #             if len(seq) > max_seq_len:
    #                 seq = seq[:max_seq_len]  # Truncate sequence if too long
    #             indices.append(df.index[i])
    #             batch_proteins.append((str(df.index[i]), seq))
    #
    #         # Convert batch into tensors
    #         _, _, batch_tokens = batch_converter(batch_proteins)
    #         batch_tokens = batch_tokens.to(device)
    #
    #         # Process each sequence in chunks if necessary
    #         with torch.no_grad():
    #             for idx, tokens in enumerate(batch_tokens):
    #                 embedding_list = []
    #                 for start in range(0, tokens.size(0), max_chunk_len):
    #                     end = min(start + max_chunk_len, tokens.size(0))
    #                     chunk_tokens = tokens[start:end].unsqueeze(0)  # Add batch dimension
    #
    #                     # Get the embedding for the chunk
    #                     results = model(chunk_tokens, repr_layers=[12], return_contacts=False)
    #                     chunk_embedding = results["representations"][12].squeeze(0)
    #
    #                     embedding_list.append(chunk_embedding.cpu().numpy())
    #
    #                 # Concatenate chunks to get the full sequence embedding
    #                 full_embedding = np.concatenate(embedding_list, axis=0)
    #
    #                 # Pad or truncate to max_seq_len
    #                 if full_embedding.shape[0] < max_seq_len:
    #                     pad_len = max_seq_len - full_embedding.shape[0]
    #                     full_embedding = np.pad(full_embedding, ((0, pad_len), (0, 0)), mode='constant')
    #                 else:
    #                     full_embedding = full_embedding[:max_seq_len, :]
    #
    #                 # Store embedding in dictionary
    #                 embeddings_dict[indices[idx]] = full_embedding
    #
    #     # Save embeddings to file
    #     np.savez(embedding_file, **embeddings_dict)
    #     print(f"Embeddings saved to {embedding_file}")

    def process_proteins_and_save_embeddings(df, embedding_file, batch_size=50, max_chunk_len=500, max_seq_len=1200):
        # Load ESM model
        model_dir = '/home/xue/htb/DrugBAN-main/ESM'
        os.environ['TORCH_HOME'] = model_dir
        print("Loading ESM model...")
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        batch_converter = alphabet.get_batch_converter()

        # Create a memory-mapped file to save embeddings
        with h5py.File(embedding_file, 'w') as h5f:
            # Initialize datasets in the file
            embedding_dim = model.embed_dim  # replace with the actual embedding dimension of the model
            embeddings = h5f.create_dataset('embeddings', (len(df), max_seq_len, embedding_dim), dtype='float32')

            # Process proteins in batches
            for start_idx in tqdm(range(0, len(df), batch_size)):
                batch_proteins = []
                indices = []

                # Prepare batch of sequences and corresponding indices
                for i in range(start_idx, min(start_idx + batch_size, len(df))):
                    seq = df.loc[i, 'Protein']
                    if len(seq) > max_seq_len:
                        seq = seq[:max_seq_len]  # Truncate sequence if too long
                    indices.append(i)
                    batch_proteins.append((str(df.index[i]), seq))

                # Convert batch into tensors
                _, _, batch_tokens = batch_converter(batch_proteins)
                batch_tokens = batch_tokens.to(device)

                # Process each sequence in chunks if necessary
                with torch.no_grad():
                    for idx, tokens in enumerate(batch_tokens):
                        embedding_list = []
                        for start in range(0, tokens.size(0), max_chunk_len):
                            end = min(start + max_chunk_len, tokens.size(0))
                            chunk_tokens = tokens[start:end].unsqueeze(0)  # Add batch dimension

                            # Get the embedding for the chunk
                            results = model(chunk_tokens, repr_layers=[12], return_contacts=False)
                            chunk_embedding = results["representations"][12].squeeze(0)

                            embedding_list.append(chunk_embedding.cpu().numpy())

                        # Concatenate chunks to get the full sequence embedding
                        full_embedding = np.concatenate(embedding_list, axis=0)

                        # Pad or truncate to max_seq_len
                        if full_embedding.shape[0] < max_seq_len:
                            pad_len = max_seq_len - full_embedding.shape[0]
                            full_embedding = np.pad(full_embedding, ((0, pad_len), (0, 0)), mode='constant')
                        else:
                            full_embedding = full_embedding[:max_seq_len, :]

                        # Store embedding in the memory-mapped file
                        embeddings[indices[idx], :, :] = full_embedding

                # Clear CUDA cache and collect garbage to free memory
                torch.cuda.empty_cache()
                gc.collect()

        print(f"Embeddings saved to {embedding_file}")

    def read_and_save_protein_embeddings(dataFolder, df_train, df_val, df_test):
        train_embedding_file = os.path.join(dataFolder, "train_protein_embeddings.h5")
        val_embedding_file = os.path.join(dataFolder, "val_protein_embeddings.h5")
        test_embedding_file = os.path.join(dataFolder, "test_protein_embeddings.h5")

        if not os.path.exists(train_embedding_file):
            print("Processing train proteins and saving embeddings...")
            process_proteins_and_save_embeddings(df_train, train_embedding_file)

        if not os.path.exists(val_embedding_file):
            print("Processing val proteins and saving embeddings...")
            process_proteins_and_save_embeddings(df_val, val_embedding_file)

        if not os.path.exists(test_embedding_file):
            print("Processing test proteins and saving embeddings...")
            process_proteins_and_save_embeddings(df_test, test_embedding_file)

        def readProteinEmbed():
            embedding_file = os.path.join(dataFolder, "protein_embeddings.pth")
            if not os.path.exists(embedding_file):
                print("Processing proteins and saving embeddings...")
                df_combined = pd.concat([df_train, df_val, df_test])
                process_proteins_and_save_embeddings(df_combined, embedding_file)

    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "val.csv")
    test_path = os.path.join(dataFolder, "test.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    read_and_save_protein_embeddings(dataFolder, df_train, df_val, df_test)
    # train_embedding_file = os.path.join(dataFolder, "train_protein_embeddings.pkl")
    # val_embedding_file = os.path.join(dataFolder, "val_protein_embeddings.pkl")
    # test_embedding_file = os.path.join(dataFolder, "test_protein_embeddings.pkl")
    #
    # train_dataset = DTIDatasetLLM(df_train.index.values, df_train, protein_file=train_embedding_file)
    # val_dataset = DTIDatasetLLM(df_val.index.values, df_val, protein_file=val_embedding_file)
    # test_dataset = DTIDatasetLLM(df_test.index.values, df_test, protein_file=test_embedding_file)


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")
