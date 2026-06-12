import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from utils.schedulers import NoamLR
from utils.utils import AA_VOCAB


class FeedForward(nn.Module):
    """
    Feed-forward network
    Args:
        d_model: int, number of input features
        d_ff: int, number of hidden features
        dropout: float, dropout rate
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class RelativePositionalEncodings(nn.Module):
    """
    Relative positional encodings
    Args:
        d_pos: int, number of positional embeddings
        max_relative: int, maximum relative position
    """

    def __init__(self, d_pos, max_relative=32):
        super().__init__()
        self.max_relative = max_relative
        self.linear = nn.Linear(2 * max_relative + 1 + 1, d_pos)
        # {-max_relative, ..., max_relative} and {unmasked}
        # mapping to {0, ..., 2 * max_relative} and {2 * max_relative + 1}

    def forward(self, offset, chain_mask):
        """
        Args:
            offset: [B, N, N], relative position of each node to its neighbors
            mask: [B, N, N], mask for nodes whether they are on the same chain

        Returns:
            E: [B, N, N, d_pos], relative positional encodings
        """
        d = torch.clip(offset + self.max_relative, 0, 2 * self.max_relative)
        d = d * chain_mask + (1 - chain_mask) * (2 * self.max_relative + 1)
        d_onehot = torch.nn.functional.one_hot(d, 2 * self.max_relative + 1 + 1)
        E = self.linear(d_onehot.float())
        return E


class StructureEmbedding(nn.Module):
    """
    Structure features Modified from `ProteinMPNN <https://doi.org/10.1126/science.add2187>`_
    Args:
        d_edge: int, number of edge features
        n_pos_emb: int, number of positional embeddings
        n_rbf: int, number of radial basis functions
        k_neighbors: int, optional, number of nearest neighbors
        dist_thres: int, optional, distance threshold for nearest neighbors
        coord_noise: float, coordinate noise
        ln_eps: float, layer norm epsilon
    """

    def __init__(
        self,
        d_edge,
        n_pos_emb=16,
        n_rbf=16,
        k_neighbors=32,
        coord_noise=0.0,
        ln_eps=1e-6,
    ):
        super().__init__()
        self.d_edge = d_edge
        self.k_neighbors = k_neighbors
        self.coord_noise = coord_noise
        self.num_rbf = n_rbf

        self.positional_encodings = RelativePositionalEncodings(n_pos_emb, k_neighbors)
        edge_in = n_pos_emb + n_rbf * 25
        self.edge_embedding = nn.Linear(edge_in, d_edge, bias=False)
        self.norm_edges = nn.LayerNorm(d_edge, eps=ln_eps)

    def _dist(self, X, mask, eps=1e-6):
        """
        Distance matrix of each node to its neighbors
        Args:
            X: [B, N, 3], node coordinates of specific atom
            mask: [B, N], mask for nodes

        Returns:
            D: [B, N, K], distance matrix of each node to its neighbors
            E_idx: [B, N, K], index of neighbors
        """
        mask_2D = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        dX = torch.unsqueeze(X, 1) - torch.unsqueeze(X, 2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1.0 - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(
            D_adjust, np.minimum(self.k_neighbors, X.shape[1]), dim=-1, largest=False
        )
        return D_neighbors, E_idx

    def _rbf(self, D):
        """
        Radial basis function
        Args:
            D: [B, N, K], distance matrix of each node to its neighbors

        Returns:
            RBF: [B, N, K, num_rbf], radial basis function based on distance
        """
        device = D.device
        D_min, D_max, D_count = 2.0, 22.0, self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
        return RBF

    def _get_rbf(self, A, B, E_idx):
        """
        Get radial basis function
        Args:
            A: [B, N, 3], node coordinates of specific atom
            B: [B, N, 3], node coordinates of another atom
            E_idx: [B, N, K], index of neighbors

        Returns:
            RBF: [B, N, K, num_rbf], radial basis function of atom A to atom B
        """
        D_A_B = torch.sqrt(
            torch.sum((A[:, :, None, :] - B[:, None, :, :]) ** 2, -1) + 1e-6
        )  # [B, N, N], distance matrix of atom A to atom B
        D_A_B_neighbors = torch.gather(
            D_A_B, -1, E_idx
        )  # [B, N, K], distance matrix of atom A to atom B of neighbor nodes
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        """
        Args:
            X: [B, N, 4, 3], node coordinates of specific atom
            mask: [B, N], mask for nodes
            residue_idx: [B, N], residue index of each node
            chain_labels: [B, N], chain label of each node

        Returns:
            h_E: [B, N, K, D], edge features
        """
        if self.training and self.coord_noise > 0:
            X = X + torch.randn_like(X) * self.coord_noise

        X_N = X[:, :, 0, :]
        X_Ca = X[:, :, 1, :]
        X_C = X[:, :, 2, :]
        X_O = X[:, :, 3, :]
        b = X_Ca - X_N
        c = X_C - X_Ca
        a = torch.cross(b, c, dim=-1)
        X_Cb = X_Ca + -0.58273431 * a + 0.56802827 * b - 0.54067466 * c

        D_neighbors, E_idx = self._dist(X_Ca, mask)

        RBF_all = []
        for A in [X_N, X_Ca, X_C, X_O, X_Cb]:
            for B in [X_N, X_Ca, X_C, X_O, X_Cb]:
                RBF_A_B = self._get_rbf(A, B, E_idx)
                RBF_all.append(RBF_A_B)
        RBF_all = torch.cat(RBF_all, dim=-1)

        offset = residue_idx[:, :, None] - residue_idx[:, None, :]
        chain_mask = (
            (chain_labels[:, :, None] - chain_labels[:, None, :]) == 0
        ).long()  # [B, N, K], mask for same chain
        offset = torch.gather(offset, -1, E_idx)  # [B, N, K]
        chain_mask = torch.gather(chain_mask, -1, E_idx)  # [B, N, K]
        positional_embedding = self.positional_encodings(
            offset, chain_mask
        )  # [B, N, K, num_positional_embeddings]

        edge = torch.cat([RBF_all, positional_embedding], dim=-1)

        h_E = self.norm_edges(self.edge_embedding(edge))  # [B, N, K, D]

        return h_E, E_idx


class EdgeEncoding(nn.Module):
    """
    Edge encoding in Graphormer Attention Bias
    Args:
        d_edge: int, dimension of edge features
        n_heads: int, number of heads in multi-head attention
    """

    def __init__(self, d_edge, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.proj = nn.Linear(d_edge, n_heads)

    def forward(self, E, E_idx):
        """
        Args:
            E: [B, N, K, D], edge features
            E_idx: [B, N, K], index of neighbors

        Returns:
            edge_bias: [B, n_heads, N, N], edge bias
        """
        B, N = E.shape[:2]
        edge_bias = self.proj(E)  # [B, N, K, n_heads]
        bias_full = torch.zeros(
            B, N, N, self.n_heads, device=E.device, dtype=edge_bias.dtype
        )
        bias_full.scatter_(
            2, E_idx.unsqueeze(-1).expand(-1, -1, -1, self.n_heads), edge_bias
        )  # [B, N, N, n_heads])
        bias_full = bias_full.permute(0, 3, 1, 2)  # [B, n_heads, N, N]
        return bias_full


class GraphormerAttention(nn.Module):
    """
    Graphormer Attention Modified from `Graphormer <https://doi.org/10.48550/arXiv.2106.05234>`_
    Args:
        d_model: int, dimension of input features
        n_heads: int, number of heads in multi-head attention
        dropout: float, dropout rate
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_head = d_model // n_heads
        self.n_heads = n_heads
        self.scale = self.d_head**-0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.dropout = nn.Dropout(dropout)
        self.edge_encoding = EdgeEncoding(d_model, n_heads)
        self.proj_o = nn.Linear(d_model, d_model)

    def forward(self, h_V, h_E, E_idx, attn_mask=None):
        """
        Args:
            h_V: [B, N, D], input features
            h_E: [B, N, K, D], edge features
            E_idx: [B, N, K], index of neighbors
            attn_mask: [B, N, N], attention mask

        Returns:
            out: [B, N, D], output features
        """
        B, N, _ = h_V.shape
        qkv = self.qkv(h_V).chunk(3, dim=-1)  # [B, N, 3*D]
        q, k, v = map(
            lambda t: t.view(B, N, self.n_heads, self.d_head).transpose(1, 2), qkv
        )  # [B, n_heads, N, d_head]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, n_heads, N, N]

        # edge bias
        edge_bias = self.edge_encoding(h_E, E_idx)  # [B, n_heads, N, N]
        scores += edge_bias

        # padding mask
        if attn_mask is not None:
            scores = scores.masked_fill(
                attn_mask.unsqueeze(1) == 0, torch.finfo(scores.dtype).min
            )

        attn = nn.functional.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        h_V = torch.matmul(attn, v)  # [B, n_heads, N, d_head]
        h_V = h_V.transpose(1, 2).contiguous().view(B, N, -1)  # [B, N, D]
        return self.proj_o(h_V)


class GraphormerBlock(nn.Module):
    """
    Graphormer Block Modified from `Graphormer <https://doi.org/10.48550/arXiv.2106.05234>`_
    Args:
        d_model: int, dimension of input features
        n_heads: int, number of heads in multi-head attention
        d_ff: int, dimension of feed-forward network
        dropout: float, dropout rate
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GraphormerAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, h_V, h_E, E_idx, attn_mask=None):
        """
        Args:
            h_V: [B, N, D], input features
            h_E: [B, N, N, D], edge features
            attn_mask: [B, N, N], attention mask
        Returns:
            out: [B, N, D], output features
        """
        h_V = self.attn(self.norm1(h_V), h_E, E_idx, attn_mask) + h_V
        h_V = self.ffn(self.norm2(h_V)) + h_V
        return h_V


class TempoNetModel(nn.Module):
    """
    TempoNet Model Structure
    Args:
        src_vocab: int, vocabulary size of input sequence
        tgt_dim: int, dimension of target prediction
        d_hidden: int, dimension of hidden features
        d_ff: int, dimension of feedforward network
        n_heads: int, number of heads in graphormer attention
        n_layers: int, number of graphormer layers
        dropout: float, dropout rate
        k_neighbors: int, number of neighbors
        coord_noise: float, noise added to coordinates
    """

    def __init__(
        self,
        src_vocab=AA_VOCAB,
        tgt_dim=1,
        d_hidden=512,
        d_ff=None,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
        k_neighbors=48,
        coord_noise=0.0,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_hidden

        self.seq_embd = nn.Embedding(src_vocab, d_hidden, padding_idx=src_vocab - 1)
        self.structure_embd = StructureEmbedding(
            d_hidden, k_neighbors=k_neighbors, coord_noise=coord_noise
        )

        self.layers = nn.ModuleList(
            [GraphormerBlock(d_hidden, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.tgt_layer = nn.Linear(d_hidden, tgt_dim)

    def forward(self, X, A, MV, IR, IC):
        """
        Args:
            X: [B, N, 3], input coordinates
            A: [B, N], input sequence
            MV: [B, N], mask valid residues
            IR: [B, N], residue index
            IC: [B, N], chain index
        Returns:
            pred: [B, N, 1], prediction of translation tempo
            embd: [B, N, D], mid embedding
        """
        h_V = self.seq_embd(A)  # [B, N, D]

        h_E, E_idx = self.structure_embd(X, MV, IR, IC)  # [B, N, K, D], [B, N, K]
        attn_mask = MV.unsqueeze(1) * MV.unsqueeze(2)  # [B, N, L]

        for layer in self.layers:
            h_V = layer(h_V, h_E, E_idx, attn_mask)  # [B, N, D]
            h_V = MV.unsqueeze(-1) * h_V  # [B, N, D]

        embd = h_V  # [B, N, D]

        pred = self.tgt_layer(embd)  # [B, N, 1]
        return pred, embd


class TempoNet(pl.LightningModule):

    def __init__(
        self,
        src_vocab=AA_VOCAB,
        tgt_dim=1,
        d_hidden=512,
        d_ff=None,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
        k_neighbors=48,
        coord_noise=0.0,
    ):
        super().__init__()
        self.model = TempoNetModel(
            src_vocab=src_vocab,
            tgt_dim=tgt_dim,
            d_hidden=d_hidden,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            k_neighbors=k_neighbors,
            coord_noise=coord_noise,
        )
        self.save_hyperparameters()

    def forward(self, X, A, MV, IR, IC):
        return self.model(X, A, MV, IR, IC)

    def _shared_step(self, batch, mode):
        X, A, F, T, IC, IR, MV, ML, NCBI_id, AFDB_id = batch
        mask = MV * ML * F / 100
        pred, _ = self.model(X, A, MV, IR, IC)

        criterion = nn.MSELoss(reduction="none")
        loss = criterion(pred.squeeze(-1), T)
        loss = loss * mask
        loss = loss.sum() / mask.sum()
        self.log(f"loss/{mode}_loss", loss)
        return loss

    def training_step(self, batch):
        return self._shared_step(batch, mode="train")

    def validation_step(self, batch):
        return self._shared_step(batch, mode="valid")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        scheduler = NoamLR(
            optimizer,
            model_size=self.hparams.d_hidden,
            warmup_steps=4000,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
