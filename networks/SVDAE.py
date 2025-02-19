import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from networks.GCN import GCN,MLP
import torch.nn.functional as F
class InnerProductDecoder(nn.Module):
    """Decoder model layer for link prediction."""
    def forward(self, z, activation=None):
        """Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())

        return activation(adj)

class SVDAE(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(SVDAE, self).__init__()
        #self.g = g
        self.A_encoder = MLP(in_feats,
                             n_hidden,
                             n_classes,
                             n_layers,
                             activation,
                             dropout)
        self.A_decoder = MLP(n_classes,
                             n_hidden,
                             in_feats,
                             n_layers,
                             activation,
                             dropout)
        self.S_encoder = GCN(g,
                             in_feats,
                             n_hidden,
                             n_classes,
                             n_layers,
                             activation,
                             dropout)
        self.Fusion = nn.ModuleList()
        self.Fusion.append( nn.Sequential(nn.Linear(n_classes*2, n_hidden),
                          nn.ReLU(inplace=True)))

        # output layer
        self.Fusion.append(nn.Linear(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.InnerProducter = InnerProductDecoder()
    def forward(self,g,features):
        # g 原图结构    adj 只有左上到右下的自定义的稀疏矩阵
        # S_z: structure   A_z: Attribute
        print("g",g)
        print("features",features)
        print("g.",g)
        print("features.shape", features.shape)
        A_z=self.A_encoder(features)
        S_z = self.S_encoder(g,features)
        print("S_z",S_z.shape)
        print("A_z",A_z.shape)
        Z = torch.add(A_z, S_z)
        print("Z",Z.shape)
        A_recon = self.A_decoder(Z)
        print("A_recon",A_recon.shape)
        S_recon=self.InnerProducter(S_z, activation=torch.sigmoid)
        print("S_recon",S_recon.shape)
        return A_z,S_z, A_recon, S_recon




