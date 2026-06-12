import torch
import torch.nn as nn
import pytorch_lightning as pl
from utils.schedulers import NoamLR
from utils.utils import AA_VOCAB, CODON_VOCAB

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TempoCoderModel(nn.Module):
    """
    TempoCoder Model Structure
    Args:
        src_vocab: int, vocabulary size of input sequence
        tgt_vocab: int, vocabulary size of target prediction
        d_hidden: int, dimension of hidden features
        d_ff: int, dimension of feedforward network
        n_heads: int, number of heads in graphormer attention
        n_layers: int, number of graphormer layers
        dropout: float, dropout rate
    """

    def __init__(
        self,
        src_vocab=AA_VOCAB,
        tgt_vocab=CODON_VOCAB,
        d_hidden=512,
        d_ff=None,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_hidden

        self.seq_embd = nn.Embedding(src_vocab, d_hidden, padding_idx=src_vocab - 1)
        self.tempo_embd = nn.Linear(1, d_hidden)
        self.src_embd = nn.Linear(d_hidden * 2, d_hidden)
        self.pos_enc = PositionalEncoding(d_hidden)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_hidden,
                    nhead=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_layers)
            ]
        )

        self.tgt_layer = nn.Linear(d_hidden, tgt_vocab)

    def forward(self, A, T, MV):
        """
        Args:
            A: [B, N], input sequence
            T: [B, N], translation tempo
            MV: [B, N], mask valid residues
        Returns:
            logits: [B, N, CODON_VOCAB], prediction of codon for each residue
            embd: [B, N, D], mid embedding
        """
        src_seq = self.seq_embd(A)
        src_tempo = self.tempo_embd(T.unsqueeze(-1))
        embd = self.src_embd(torch.cat([src_seq, src_tempo], dim=-1))
        embd = self.pos_enc(embd)
        for layer in self.layers:
            embd = layer(embd, src_key_padding_mask=1 - MV)
        logits = self.tgt_layer(embd)
        return logits, embd


class TempoCoder(pl.LightningModule):
    def __init__(
        self,
        src_vocab=AA_VOCAB,
        tgt_vocab=CODON_VOCAB,
        d_hidden=512,
        d_ff=None,
        n_heads=4,
        n_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.model = TempoCoderModel(
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            d_hidden=d_hidden,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.save_hyperparameters()

    def forward(self, A, T, MV):
        return self.model(A, T, MV)

    def _shared_step(self, batch, mode):
        A, C, T, MV, ML, NCBI_id = batch
        mask = MV * ML
        logits, _ = self.model(A, T, MV)

        criterion = nn.CrossEntropyLoss(reduction="none")
        loss = (
            criterion(
                logits.contiguous().view(-1, logits.size(-1)), C.view(-1)
            ).view_as(C)
            * mask
        )
        loss = loss.sum() / mask.sum()

        acc = (logits.argmax(dim=-1) == C).float() * mask
        acc = acc.sum() / mask.sum()
        self.log(f"loss/{mode}_loss", loss)
        self.log(f"loss/{mode}_acc", acc)
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

    def sample_codons(self, A, T, MV, n_samples=3, temperature=1.0):
        """
        Sample codon sequences from the model's predictions given input sequence and translation tempo.
        Args:
            A: [B, N], input sequence
            T: [B, N], translation tempo
            MV: [B, N], mask valid residues
            n_samples: int, number of codon sequences to sample for each input
            temperature: float, sampling temperature
        Returns:
            codons: [n_samples, B, N], sampled codon sequences
            confidences: [n_samples, B, N], confidence of the sampled codons
        """
        logit, _ = self.model(A, T, MV)  # [B, N, tgt_vocab]
        probs = torch.softmax(logit / temperature, dim=-1)

        codons = []
        confidences = []
        for _ in range(n_samples):
            # Sample codon for each residue from the predicted distribution
            codon = torch.multinomial(
                probs.view(-1, logit.size(-1)), num_samples=1
            ).view(A.shape[0], A.shape[1])

            # Get the confidence of the sampled codon
            conf = probs.gather(-1, codon.unsqueeze(-1)).squeeze(-1)

            codons.append(codon)
            confidences.append(conf)

        return torch.stack(codons), torch.stack(confidences)
