import torch
import pickle

# fold_list: List[List[List[Tuple[int, int, List[int], List[int]]]]]], all 10 folds
# len(fold_list) == 10, the same as fold_count
#
# fold_list[i]: List[List[Tuple[int, int, List[int], List[int]]]], the i-th fold
# len(fold_list[i]) == cluster_count, the same as the number of clusters in fold i
#
# fold_list[i][j]: List[Tuple[int, int, List[int], List[int]]], the j-th cluster in fold i
# len(fold_list[i][j]) == number of proteins in cluster j in fold i
#
# fold_list[i][j][k]: Tuple[int, int, List[int], List[int]], the k-th protein in cluster j in fold i
# fold_list[i][j][k][0]: int, the id of protein k in cluster j in fold i
# fold_list[i][j][k][1]: int, the length of protein k in cluster j in fold i
# fold_list[i][j][k][2]: List[int], NCBI ids with the same seq as protein k in cluster j in fold i
# fold_list[i][j][k][3]: List[int], AFDB ids with the same seq as protein k in cluster j in fold i
#
# ncbi_id = fold_list[i][j][k][2][l]
# ncbi_path = os.path.join(NCBI_dir, f"NCBI_{ncbi_id}.pkl")
# ncbi_data = pickle.load(open(ncbi_path, "rb"))
# ncbi_data = {
#     "codon": np.array(codon_array, dtype=np.int32),
#     "aa": np.array(aa_array, dtype=np.int32),
#     "rate": np.array(rate, dtype=np.float32),
# }
#
# afdb_id = fold_list[i][j][k][3][l]
# afdb_path = os.path.join(AFDB_dir, f"AFDB_{afdb_id}.pkl")
# afdb_data = pickle.load(open(afdb_path, "rb"))
# afdb_data = {
#     "xyz": np.array(xyz, dtype=np.float32),
#     "chain_idx": np.array(chain_idx, dtype=np.int32),
#     "b_factor": np.array(b_factors, dtype=np.float32),
#     "aa": aa_array = np.array(aa_array, dtype=np.int32),
# }


class Entry:
    def __init__(self, entry: tuple[int, int, list[int], list[int]]):
        self.id = entry[0]
        self.length = entry[1]
        self.ncbi_ids = entry[2]
        self.afdb_ids = entry[3]


class Cluster:
    def __init__(self, cluster: list[tuple[int, int, list[int], list[int]]]):
        self.entries = [Entry(entry) for entry in cluster]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


class TempoDataset(torch.utils.data.Dataset):
    def __init__(self, fold_list_pkl_path: str, folds: list[int]):
        super().__init__()
        with open(fold_list_pkl_path, "rb") as f:
            fold_list = pickle.load(f)
        self.data = []
        for fold in folds:
            self.data += [Cluster(cluster) for cluster in fold_list[fold]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
