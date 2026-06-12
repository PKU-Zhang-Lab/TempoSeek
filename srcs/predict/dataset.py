"""
Inference datasets and dataloaders for trained models.
"""

import os
import torch
import pickle
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from utils.utils import AA_VOCAB, aa_alphabet
from tqdm import tqdm

# Suppress Bio.PDB noisy numpy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="Bio.PDB")

class TempoNetPredictDataset(torch.utils.data.Dataset):
    """Load PDB/CIF files and prepare for TempoNet inference.

    Args:
        files: List of (input_pdb, output_csv, pkl_cache) tuples
        quiet: Suppress tqdm progress bar
    """

    def __init__(self, files: list[tuple[str, str, str]], quiet: bool = False):
        super().__init__()
        import gzip
        from Bio.PDB import PDBParser, FastMMCIFParser, is_aa
        from Bio.Data.PDBData import protein_letters_3to1_extended


        def _load_or_parse(pdb_path: str, pkl_path: str):
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    return pickle.load(f)
            pdb_path = str(pdb_path)
            gz = pdb_path.endswith(".gz")
            tp = ".pdb" if ".pdb" in pdb_path else ".cif"
            if gz:
                f = gzip.open(pdb_path, "rt")
            else:
                f = open(pdb_path, "r")
            parser = PDBParser(QUIET=True) if tp == ".pdb" else FastMMCIFParser(QUIET=True)
            structures = parser.get_structure("x", f)
            f.close()
            _xyz, chain_idx, aa_array, residue_seq = [], [], [], []
            chain_name = []  # actual chain IDs from PDB
            for sidx, struct in enumerate(structures):
                struct.atom_to_internal_coordinates()
                for chain in struct.get_chains():
                    cid = chain.get_id()
                    residues = []
                    for res in chain.get_residues():
                        if not is_aa(res):
                            continue
                        sn = res.get_id()[1]
                        xyz_r = np.array([
                            res["N"].coord if "N" in res else np.full(3, np.nan),
                            res["CA"].coord if "CA" in res else np.full(3, np.nan),
                            res["C"].coord if "C" in res else np.full(3, np.nan),
                            res["O"].coord if "O" in res else np.full(3, np.nan),
                        ], dtype=np.float32)
                        aa_char = protein_letters_3to1_extended.get(res.resname, "X")
                        residues.append((sn, xyz_r, aa_char))
                    residues.sort(key=lambda r: r[0])
                    prev = None
                    for sn, xyz_r, aa_char in residues:
                        if prev is not None and sn > prev + 1:
                            for g in range(prev + 1, sn):
                                _xyz.append(np.full((4, 3), np.nan))
                                chain_idx.append(sidx)
                                chain_name.append(cid)
                                aa_array.append(-1)
                                residue_seq.append(g)
                        _xyz.append(xyz_r)
                        chain_idx.append(sidx)
                        chain_name.append(cid)
                        aa_array.append(-1 if aa_char not in aa_alphabet
                                        else aa_alphabet.index(aa_char))
                        residue_seq.append(sn)
                        prev = sn
            if len(_xyz) == 0:
                raise ValueError(f"No valid residues in {pdb_path}")
            d = {
                "xyz": np.array(_xyz, dtype=np.float32),
                "chain_idx": np.array(chain_idx, dtype=np.int32),
                "aa": np.array(aa_array, dtype=np.int32),
                "res_idx": np.array(residue_seq, dtype=np.int32),
                "chain_name": chain_name,
            }
            os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
            with open(pkl_path, "wb") as f:
                pickle.dump(d, f)
            return d

        self.entries = []
        for inp, out, pkl in tqdm(files, desc="Loading structures", disable=quiet):
            d = _load_or_parse(inp, pkl)
            self.entries.append((len(d["aa"]), pkl, out))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


class TempoNetPredictDataLoader:
    """Batch PDB/CIF files for TempoNet inference.

    Args:
        files: List of (input_pdb, output_csv, pkl_cache) tuples
        batch_size: Token limit per batch
        device: torch device
        quiet: Suppress tqdm progress bar
    """

    def __init__(
        self,
        files: list[tuple[str, str, str]],
        batch_size: int,
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        quiet: bool = False,
    ):
        self.dataset = TempoNetPredictDataset(files, quiet=quiet)
        self.batch_size = batch_size
        self.device = device
        self.buckets = []
        self.resample_buckets()

    def resample_buckets(self):
        # randomly select a protein from each cluster
        lengths = [i[0] for i in self.dataset]
        sorted_indices = np.argsort(lengths)
        # greedy put clusters into buckets
        self.buckets = []
        cur_bucket, bucket_cnt = [], 0
        for idx in sorted_indices:
            L = lengths[idx]
            if cur_bucket and (bucket_cnt + 1) * L > self.batch_size:
                self.buckets.append(cur_bucket)
                cur_bucket = []
                bucket_cnt = 0
            cur_bucket.append(idx)
            bucket_cnt += 1
        if cur_bucket:  # last bucket
            self.buckets.append(cur_bucket)

    def __iter__(self):
        for bucket in self.buckets:
            batch = [self.dataset[i] for i in bucket]
            yield self.featurize(batch)

    def __len__(self):
        return len(self.buckets)

    def featurize(self, batch: list[tuple[int, str, str]]):
        # batch[i] = (seq_len, pkl_path, output_path)
        B = len(batch)  # batch size
        N = max([b[0] for b in batch])  # max seq length

        X = np.ones([B, N, 4, 3]) * np.nan  # coordinates of N, Ca, C, O atoms
        A = np.ones([B, N], dtype=np.int32)  # sequence of amino acids
        IC = np.ones([B, N], dtype=np.int32)  # chain index
        IR = np.ones([B, N], dtype=np.int32)  # residue index
        MV = np.zeros([B, N], dtype=np.int32)  # mask of valid tokens
        ResNo = np.zeros([B, N], dtype=np.int32)  # residue sequence numbers
        Len = np.zeros([B], dtype=np.int32)  # original length of each protein
        Name = []  # name of each protein in batch
        Out = []  # output CSV path for each protein
        ChainName = []  # chain name for each protein in batch

        A = A * (AA_VOCAB - 1)  # AA_VOCAB - 1 means padding
        IC = IC * -1  # -1 means no chain
        IR = IR * -1000  # -1000 means no residue

        for i, (seq_len, pkl_path, out_path) in enumerate(batch):
            Len[i] = seq_len
            Out.append(out_path)
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            X[i, :seq_len] = data["xyz"]
            A[i, :seq_len] = data["aa"]
            A[A == -1] = AA_VOCAB - 2  # unknown set to AA_VOCAB - 2

            IC[i, :seq_len] = data["chain_idx"]
            ResNo[i, :seq_len] = data["res_idx"]
            chain_idx_unique, chain_idx_first_idx = np.unique(
                data["chain_idx"], return_index=True
            )  # unique chain_idx and their first appear index
            chain_len_list = np.bincount(data["chain_idx"])[
                chain_idx_unique[np.argsort(chain_idx_first_idx)]
            ]  # chain length list regardless of chain_idx but according to their appearing order
            IR[i, :seq_len] = np.hstack(
                [np.arange(a) for a in chain_len_list]
            ) + 1000 * np.array(
                data["chain_idx"]
            )  # 1000 * chain_idx + residue_idx
            Name.append(Path(pkl_path).stem)
            ChainName.append(data["chain_name"])

        MV = np.isfinite(np.sum(X, axis=(2, 3))).astype(np.int32)
        X[np.isnan(X)] = 0
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        A = torch.tensor(A, dtype=torch.int32).to(self.device)
        IC = torch.tensor(IC, dtype=torch.int32).to(self.device)
        IR = torch.tensor(IR, dtype=torch.int32).to(self.device)
        MV = torch.tensor(MV, dtype=torch.int32).to(self.device)
        ResNo = torch.tensor(ResNo, dtype=torch.int32).to(self.device)
        Len = torch.tensor(Len, dtype=torch.int32).to(self.device)
        return X, A, IC, IR, MV, ResNo, Len, Name, Out, ChainName


#################################
#  TempoCoderPredictDataset    #
#################################
class TempoCoderPredictDataset(torch.utils.data.Dataset):
    """Read TempoNet CSV outputs and prepare for TempoCoder inference.

    Args:
        csv_path: Single CSV file or directory of CSV files
        quiet: Suppress tqdm progress bar
    """

    _AA_MAP = {c: i for i, c in enumerate(aa_alphabet)}

    def __init__(self, files: list[tuple[str, str]], quiet: bool = False):
        super().__init__()
        self.entries = []
        for inp, out in tqdm(files, desc="Checking CSVs", disable=quiet):
            df = pd.read_csv(inp, usecols=["aa", "tempo", "chain"])
            if len(df) == 0:
                raise ValueError(f"Empty CSV: {inp}")
            aa_len = df["aa"].dropna().shape[0]
            tempo_len = df["tempo"].dropna().shape[0]
            if aa_len != tempo_len:
                raise ValueError(
                    f"Column length mismatch in {inp}: "
                    f"aa={aa_len}, tempo={tempo_len}"
                )

            out_path = Path(out)
            if "chain" in df.columns:
                # Split by chain: one entry per chain
                for chain_id, grp in df.groupby("chain"):
                    chain_len = len(grp)
                    chain_out = str(out_path.with_stem(f"{out_path.stem}_Chain_{chain_id}"))
                    self.entries.append((chain_len, inp, chain_out, chain_id))
            else:
                self.entries.append((aa_len, inp, out, None))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


class TempoCoderPredictDataLoader:
    """Batch CSV data for TempoCoder inference with greedy bin-packing.

    Args:
        files: List of (input_csv, output_csv) tuples
        batch_size: Token limit per batch
        device: torch device
        quiet: Suppress tqdm progress bar
    """

    def __init__(
        self,
        files: list[tuple[str, str]],
        batch_size: int = 5000,
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        quiet: bool = False,
    ):
        self.dataset = TempoCoderPredictDataset(files, quiet=quiet)
        self.batch_size = batch_size
        self.device = device
        self.buckets = []
        self.resample_buckets()

    def resample_buckets(self):
        # greedy bin-packing by sequence length
        lengths = [e[0] for e in self.dataset.entries]
        sorted_indices = np.argsort(lengths)
        self.buckets = []
        cur_bucket, bucket_cnt = [], 0
        for idx in sorted_indices:
            L = lengths[idx]
            if cur_bucket and (bucket_cnt + 1) * L > self.batch_size:
                self.buckets.append(cur_bucket)
                cur_bucket = []
                bucket_cnt = 0
            cur_bucket.append(idx)
            bucket_cnt += 1
        if cur_bucket:  # last bucket
            self.buckets.append(cur_bucket)

    def __len__(self):
        return len(self.buckets)

    def __iter__(self):
        for bucket in self.buckets:
            yield self.featurize(bucket)

    def featurize(self, indices: list[int]):
        entries = [self.dataset[i] for i in indices]
        B = len(entries)
        N = max(e[0] for e in entries)

        A = np.full((B, N), AA_VOCAB - 1, dtype=np.int32)
        T = np.zeros((B, N), dtype=np.float32)
        MV = np.zeros((B, N), dtype=np.int32)
        RS = np.zeros((B, N), dtype=np.int32)
        names = []
        outs = []
        rs_valid = []

        for i, (length, inp, out, chain_id) in enumerate(entries):
            df = pd.read_csv(inp)
            if chain_id is not None:
                df = df[df["chain"] == chain_id].reset_index(drop=True)
            aa_idx = df["aa"].map(self.dataset._AA_MAP).fillna(AA_VOCAB - 2).astype(np.int32).values
            A[i, :length] = aa_idx
            T[i, :length] = df["tempo"].values.astype(np.float32)
            MV[i, :length] = 1
            has_rs = "res_idx" in df.columns and len(df) == length
            if has_rs:
                RS[i, :length] = df["res_idx"].values.astype(np.int32)
            else:
                RS[i, :length] = np.arange(1, length + 1, dtype=np.int32)
            rs_valid.append(has_rs)
            names.append(Path(inp).stem)
            outs.append(out)

        A = torch.tensor(A, dtype=torch.long, device=self.device)
        T = torch.tensor(T, dtype=torch.float32, device=self.device)
        MV = torch.tensor(MV, dtype=torch.float32, device=self.device)
        RS = torch.tensor(RS, dtype=torch.int32, device=self.device)

        return A, T, MV, RS, names, outs, rs_valid
