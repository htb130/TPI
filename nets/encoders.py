import torch
import torch.nn as nn
import torch.nn.functional as F
from dgllife.model.gnn import GCN, GAT, GIN, GraphSAGE

class ProteinLLMCNN(nn.Module):
	def __init__(self, embedding_dim, num_filters, kernel_size):
		super(ProteinLLMCNN, self).__init__()
		in_ch = [embedding_dim] + num_filters
		self.in_ch = in_ch[-1]
		kernels = kernel_size
		self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0], padding='same')
		self.bn1 = nn.BatchNorm1d(in_ch[1])
		self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1], padding='same')
		self.bn2 = nn.BatchNorm1d(in_ch[2])
		self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2], padding='same')
		self.bn3 = nn.BatchNorm1d(in_ch[3])

	def forward(self, v):
		# v = self.embedding(v.long())
		v = v.transpose(2, 1)
		v = self.bn1(F.relu(self.conv1(v)))
		v = self.bn2(F.relu(self.conv2(v)))
		v = self.bn3(F.relu(self.conv3(v)))
		v = v.view(v.size(0), v.size(2), -1)
		return v



# class ProteinCNN(nn.Module):
# 	def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
# 		super(ProteinCNN, self).__init__()
# 		if padding:
# 			self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
# 		else:
# 			self.embedding = nn.Embedding(26, embedding_dim)
# 		in_ch = [embedding_dim] + num_filters
# 		self.in_ch = in_ch[-1]
# 		kernels = kernel_size
# 		self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
# 		self.bn1 = nn.BatchNorm1d(in_ch[1])
# 		self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
# 		self.bn2 = nn.BatchNorm1d(in_ch[2])
# 		self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
# 		self.bn3 = nn.BatchNorm1d(in_ch[3])
#
# 	def forward(self, v):
# 		v = self.embedding(v.long())
# 		v = v.transpose(2, 1)
# 		v = self.bn1(F.relu(self.conv1(v)))
# 		v = self.bn2(F.relu(self.conv2(v)))
# 		v = self.bn3(F.relu(self.conv3(v)))
# 		v = v.view(v.size(0), v.size(2), -1)
# 		return v


class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0], padding='same')
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1], padding='same')
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2], padding='same')
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)  # [batch_size, embedding_dim, seq_len]
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.transpose(2, 1)
        return v



class VGAE(nn.Module):
	def __init__(self, encoder, latent_dim, batch_size=64):
		super(VGAE, self).__init__()
		self.batch_size = batch_size
		self.encoder = encoder
		self.mean_layer = nn.Linear(encoder.output_feats, latent_dim)
		self.log_std_layer = nn.Linear(encoder.output_feats, latent_dim)

	def forward(self, batch_graph):
		device = next(self.parameters()).device
		node_feats = self.encoder(batch_graph)

		mean = self.mean_layer(node_feats)
		log_std = self.log_std_layer(node_feats)
		std = torch.exp(log_std)

		z = mean + std * torch.randn_like(std)

		batch_num_nodes = batch_graph.batch_num_nodes().tolist()
		batch_size = len(batch_num_nodes)

		adj_original = batch_graph.adjacency_matrix(transpose=True, scipy_fmt="csr").todense()
		adj_original = torch.tensor(adj_original, dtype=torch.float32).to(device) # lb: PyTorch  Ï

		re_losses = []
		node_start = 0
		for i in range(batch_size):
			num_nodes = batch_num_nodes[i]
			adj_original_i = adj_original[node_start:node_start + num_nodes, node_start:node_start + num_nodes]
			adj_reconstructed = torch.sigmoid(torch.matmul(z[i], z[i].transpose(-1, -2)))
			re_loss = self.loss_function(adj_reconstructed, adj_original_i, mean[i], log_std[i])
			re_losses.append(re_loss)
			node_start += num_nodes

		re_loss = torch.mean(torch.stack(re_losses))

		return node_feats, z, re_loss

	def loss_function(self, adj_reconstructed, adj_original, mean, log_std):
		recon_loss = F.binary_cross_entropy(adj_reconstructed, adj_original)
		kl_divergence = -0.5 * torch.mean(1 + 2 * log_std - mean ** 2 - torch.exp(2 * log_std))
		return recon_loss + kl_divergence


class MolecularGCN(nn.Module):
	def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
		super(MolecularGCN, self).__init__()
		self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
		if padding:
			with torch.no_grad():
				self.init_transform.weight[-1].fill_(0)
		self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
		self.output_feats = hidden_feats[-1]

	def forward(self, batch_graph):
		node_feats = batch_graph.ndata.pop('h')
		node_feats = self.init_transform(node_feats)
		node_feats = self.gnn(batch_graph, node_feats)
		batch_size = batch_graph.batch_size
		node_feats = node_feats.view(batch_size, -1, self.output_feats)
		return node_feats





