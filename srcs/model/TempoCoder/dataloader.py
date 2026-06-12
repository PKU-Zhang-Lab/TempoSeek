import os
import torch
import pickle
import numpy as np
from utils.dataset import TempoDataset, Entry
from utils.utils import AA_VOCAB, CODON_VOCAB

# Dataset[i] = Cluster
# Cluster[j] = Entry
# Entry has the following attributes:
# - id: int, the id of the protein
# - length: int, the length of the protein
# - ncbi_ids: List[int], the list of NCBI ids with the same sequence as the protein
# - afdb_ids: List[int], the list of AFDB ids with the same sequence as the protein

class TempoCoderDataLoader:

    def __init__(
        self,
        fold_list_pkl_path: str,
        folds: list[int],
        batch_size: int,  # HERE batch size means total token limit
        ncbi_dir: str = "data/NCBI_pkl",  # NCBI data dir
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
        self.check_files_sanity = check_files_sanity
        self.aa_vocab_size = AA_VOCAB
        self.codon_vocab_size = CODON_VOCAB

        self.check_files_exist()

        self.prot_indices: list[int] = []
        # prot_indices[i] = index of protein selected in cluster i
        self.buckets: list[list[int]] = []
        # buckets[i] = indices of clusters in bucket i

        self.resample_buckets()

    def check_files_exist(self):
        if not os.path.exists(self.NCBI_dir):
            raise FileNotFoundError(f"NCBI dir {self.NCBI_dir} not found")
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
        # batch[i] = Entry(seq_len, ncbi_ids, afdb_ids)
        B = len(batch)  # batch size
        N = max([b.length for b in batch])  # max seq length

        A = np.ones([B, N], dtype=np.int32)  # sequence of amino acids
        C = np.ones([B, N], dtype=np.int32)  # sequence of codons
        T = np.ones([B, N], dtype=np.float32)  # translation tempo
        MV = np.zeros([B, N], dtype=np.int32)  # mask of valid tokens
        ML = np.zeros([B, N], dtype=np.int32)  # mask of loss
        NCBI_id = np.zeros([B], dtype=np.int32)  # NCBI id

        A = A * (self.aa_vocab_size - 1)  # AA_VOCAB - 1 means padding
        C = C * (self.codon_vocab_size - 1)  # CODON_VOCAB - 1 means padding
        T = T * np.nan  # nan means padding
        for i, entry in enumerate(batch):
            seq_len = entry.length
            ncbi_ids = entry.ncbi_ids
            ncbi_id = np.random.choice(ncbi_ids)
            with open(os.path.join(self.NCBI_dir, f"NCBI_{ncbi_id}.pkl"), "rb") as f:
                ncbi_data = pickle.load(f)
            A[i, :seq_len] = ncbi_data["aa"]
            C[i, :seq_len] = ncbi_data["codon"][:-1]
            T[i, :seq_len] = ncbi_data["tempo"]

            if self.mask_all:
                mask = np.ones(seq_len)
            else:
                raise NotImplementedError("masking not implemented yet")
            ML[i, :seq_len] = mask
            NCBI_id[i] = ncbi_id
        A[A == -1] = self.aa_vocab_size - 2  # unknown set to AA_VOCAB - 2
        C[C == -1] = self.codon_vocab_size - 2  # unknown set to CODON_VOCAB - 2
        MV = np.isfinite(T).astype(np.int32)
        T[np.isnan(T)] = 0

        A = torch.from_numpy(A).to(dtype=torch.long, device=self.device)
        C = torch.from_numpy(C).to(dtype=torch.long, device=self.device)
        T = torch.from_numpy(T).to(dtype=torch.float32, device=self.device)
        MV = torch.from_numpy(MV).to(dtype=torch.float32, device=self.device)
        ML = torch.from_numpy(ML).to(dtype=torch.float32, device=self.device)
        NCBI_id = torch.from_numpy(NCBI_id).to(dtype=torch.long, device=self.device)

        return A, C, T, MV, ML, NCBI_id