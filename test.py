# import pickle
# protein_file = './datasets/bindingdb/random/train_protein_embeddings.pkl'
# with open(protein_file, 'rb') as f:
#     protein_embeddings = pickle.load(f)
# print("Protein LLM Embeddings loadded")
# print(protein_embeddings[2].shape)

# import h5py
# embedding_file = './datasets/bindingdb/random/train_protein_embeddings.h5'
# with h5py.File(embedding_file, 'r') as h5f:
#     embeddings = h5f['embeddings']
#
#     protein_index = 1589
#     embedding_data = embeddings[protein_index]
#     print(f"Embedding for protein len:", len(embeddings))
#     print(f"Embedding for protein at index {protein_index}:", embedding_data.shape)

import numpy as np
import pandas as pd

p = np.load('data/celegans/proteinsembeddings.npy', allow_pickle=True)

p_resized = p[:, 0, 1:-1, :]
print("Resized shape:", p_resized.shape)

np.save('data/celegans/proteinsembeddings.npy', p_resized)
print("##---finished---##")