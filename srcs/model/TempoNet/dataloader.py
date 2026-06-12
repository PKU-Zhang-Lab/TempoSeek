import os
import torch
import pickle
import numpy as np
from utils.dataset import TempoDataset, Entry
from utils.utils import AA_VOCAB

# Dataset[i] = Cluster
# Cluster[j] = Entry
# Entry has the following attributes:
# - id: int, the id of the protein
# - length: int, the length of the protein
# - ncbi_ids: List[int], the list of NCBI ids with the same sequence as the protein
# - afdb_ids: List[int], the list of AFDB ids with the same sequence as the protein

class TempoNetDataLoader:
    def __init__(
        self,
        fold_list_pkl_path: str,
        folds: list[int],
        batch_size: int,  # HERE batch size means total token limit
        ncbi_dir: str = "data/NCBI_pkl",  # NCBI data dir
        afdb_dir: str = "data/AFDB_pkl",  # AFDB data dir
        device: torch.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
        shuffle: bool = True,  # shuffle while iterating
        resample: bool = True,  # resample while finishing an epoch
        mask_all: bool = True,  # mask all tokens
        check_files_sanity: bool = False,  # check if all files readable during initialization, which can be time consuming but can catch errors early
    ):
        self.dataset = TempoDataset(fold_list_pkl_path, folds)
        self.batch_size = batch_size
        self.mask_all = mask_all
        self.device = device
        self.shuffle = shuffle
        self.resample = resample
        self.NCBI_dir = ncbi_dir
        self.AFDB_dir = afdb_dir
        self.check_files_sanity = check_files_sanity
        self.aa_vocab_size = AA_VOCAB

        self.check_files_exist()

        self.prot_indices: list[int] = []
        # prot_indices[i] = index of protein selected in cluster i
        self.buckets: list[list[int]] = []
        # buckets[i] = indices of clusters in bucket i

        self.resample_buckets()

    def check_files_exist(self):
        if not os.path.exists(self.NCBI_dir):
            raise FileNotFoundError(f"NCBI dir {self.NCBI_dir} not found")
        if not os.path.exists(self.AFDB_dir):
            raise FileNotFoundError(f"AFDB dir {self.AFDB_dir} not found")
        for cluster in self.dataset:
            for entry in cluster:
                for n_id in entry.ncbi_ids:
                    n_path = os.path.join(self.NCBI_dir, f"NCBI_{n_id}.pkl")
                    if not os.path.exists(n_path):
                        raise FileNotFoundError(
                            f"NCBI file NCBI_{n_id}.pkl not found in {self.NCBI_dir}"
                        )
                    if self.check_files_sanity:
                        with open(n_path, "rb") as f:
                            pickle.load(f)
                for a_id in entry.afdb_ids:
                    a_path = os.path.join(self.AFDB_dir, f"AFDB_{a_id}.pkl")
                    if not os.path.exists(a_path):
                        raise FileNotFoundError(
                            f"AFDB file AFDB_{a_id}.pkl not found in {self.AFDB_dir}"
                        )
                    if self.check_files_sanity:
                        with open(a_path, "rb") as f:
                            pickle.load(f)

    def resample_buckets(self):
        # randomly select a protein from each cluster
        self.prot_indices = []
        lengths = []
        for i in range(len(self.dataset)):
            cluster = self.dataset[i]
            index = np.random.randint(len(cluster))
            self.prot_indices.append(index)
            lengths.append(cluster[index].length)
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

    def __len__(self):
        return len(self.buckets)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.buckets)
        for bucket in self.buckets:
            batch = [self.dataset[i][self.prot_indices[i]] for i in bucket]
            yield self.featurize(batch)
        if self.resample:
            self.resample_buckets()

    def featurize(self, batch: list[Entry]):
        # batch[i] = (seq_len, ncbi_ids, afdb_ids)
        B = len(batch)  # batch size
        N = max([b.length for b in batch])  # max seq length

        X = np.ones([B, N, 4, 3]) * np.nan  # coordinates of N, Ca, C, O atoms
        A = np.ones([B, N], dtype=np.int32)  # sequence of amino acids
        F = np.zeros([B, N], dtype=np.float32)  # b factors
        T = np.zeros([B, N], dtype=np.float32)  # translation tempo
        IC = np.ones([B, N], dtype=np.int32)  # chain index
        IR = np.ones([B, N], dtype=np.int32)  # residue index
        MV = np.zeros([B, N], dtype=np.int32)  # mask of valid tokens
        ML = np.zeros([B, N], dtype=np.int32)  # mask of loss
        NCBI_id = np.zeros([B], dtype=np.int32)  # NCBI id
        AFDB_id = np.zeros([B], dtype=np.int32)  # AFDB id

        A = A * (self.aa_vocab_size - 1)  # AA_VOCAB - 1 means padding
        IC = IC * -1  # -1 means no chain
        IR = IR * -1000  # -1000 means no residue

        for i, entry in enumerate(batch):
            seq_len = entry.length
            ncbi_ids = entry.ncbi_ids
            afdb_ids = entry.afdb_ids
            ncbi_id = np.random.choice(ncbi_ids)
            afdb_id = np.random.choice(afdb_ids)
            with open(os.path.join(self.NCBI_dir, f"NCBI_{ncbi_id}.pkl"), "rb") as f:
                ncbi_data = pickle.load(f)
            with open(os.path.join(self.AFDB_dir, f"AFDB_{afdb_id}.pkl"), "rb") as f:
                afdb_data = pickle.load(f)
            X[i, :seq_len] = afdb_data["xyz"]
            A[i, :seq_len] = ncbi_data["aa"]
            F[i, :seq_len] = afdb_data["b_factor"]
            T[i, :seq_len] = ncbi_data["tempo"]

            IC[i, :seq_len] = afdb_data["chain_idx"]
            chain_idx_unique, chain_idx_first_idx = np.unique(
                afdb_data["chain_idx"], return_index=True
            )  # unique chain_idx and their first appear index
            chain_len_list = np.bincount(afdb_data["chain_idx"])[
                chain_idx_unique[np.argsort(chain_idx_first_idx)]
            ]  # chain length list regardless of chain_idx but according to their appearing order
            IR[i, :seq_len] = np.hstack(
                [np.arange(a) for a in chain_len_list]
            ) + 1000 * np.array(
                afdb_data["chain_idx"]
            )  # 1000 * chain_idx + residue_idx

            if self.mask_all:
                mask = np.ones(seq_len)
            else:
                raise NotImplementedError("masking not implemented yet")
                # Here we can implement loss masking based on some threshold or other criteria
            ML[i, :seq_len] = mask
            NCBI_id[i] = ncbi_id
            AFDB_id[i] = afdb_id
        A[A == -1] = self.aa_vocab_size - 2  # unknown set to AA_VOCAB - 2
        MV = np.isfinite(np.sum(X, axis=(2, 3))).astype(np.int32)
        X[np.isnan(X)] = 0

        X = torch.from_numpy(X).to(dtype=torch.float32, device=self.device)
        A = torch.from_numpy(A).to(dtype=torch.long, device=self.device)
        F = torch.from_numpy(F).to(dtype=torch.float32, device=self.device)
        T = torch.from_numpy(T).to(dtype=torch.float32, device=self.device)
        IC = torch.from_numpy(IC).to(dtype=torch.long, device=self.device)
        IR = torch.from_numpy(IR).to(dtype=torch.long, device=self.device)
        MV = torch.from_numpy(MV).to(dtype=torch.float32, device=self.device)
        ML = torch.from_numpy(ML).to(dtype=torch.float32, device=self.device)
        NCBI_id = torch.from_numpy(NCBI_id).to(dtype=torch.long, device=self.device)
        AFDB_id = torch.from_numpy(AFDB_id).to(dtype=torch.long, device=self.device)

        return X, A, F, T, IC, IR, MV, ML, NCBI_id, AFDB_id
