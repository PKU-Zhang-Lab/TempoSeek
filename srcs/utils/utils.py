import numpy as np
from Bio.Seq import Seq
import pandas as pd
import os

AA_VOCAB = 20 + 1 + 1  # 20 amino acids, 1 unknown, 1 padding
CODON_VOCAB = 64 + 1 + 1  # 64 codons, 1 unknown, 1 padding

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
info_dir = os.path.join(base_dir, "info")
aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"  # Amino Acids
dna_base = "ACGT"
codon_list = ["".join([a, b, c]) for a in dna_base for b in dna_base for c in dna_base]


def tempo_calculator(
    DNA: str,
    codon_table: pd.DataFrame = None,
) -> list:
    if codon_table is None:
        codon_table = pd.read_csv(os.path.join(info_dir, "codon.csv"))
    """Calculate the translation tempo of a mRNA sequence"""
    seq = DNA
    length = int(len(seq) / 3)

    # Devide mRNA to codons
    codon_seq = [seq[3 * i : 3 * i + 3] for i in range(length)]
    if Seq(codon_seq[-1]).translate() == "*":
        length -= 1
    codon_seq = codon_seq[:length]

    # Calculate basic k and basic translation tempo of codons
    basic_t = []  # Basic translation tempo
    for i in codon_seq:
        k = codon_table[codon_table["codon"] == i]["k"]
        if len(k) == 1:
            k = k.values[0]
            basic_t.append(1 / k)
        else:
            basic_t.append(np.nan)
    basic_t = np.array(basic_t)
    # print(basic_t)
    basic_t_series = pd.Series(basic_t)
    basic_t_interpolated = basic_t_series.interpolate(method="linear").values
    # print(basic_t_interpolated)
    # Calculate translation tempo of codons
    tempo = []  # Translation tempo
    for i in range(len(basic_t_interpolated)):
        tmp = []
        for j in range(-9, 10):
            if i + j < 0 or i + j >= length:
                continue
            tmp.append(basic_t_interpolated[i + j])
        tempo.append(sum(tmp) / len(tmp))

    return tempo

# codon_table = pd.read_csv(os.path.join(info_dir, "codon.csv"))
# tRNA_table = pd.read_csv(os.path.join(info_dir, "tRNA.csv"))
# tRNA_codon_table = (
#     tRNA_table[tRNA_table["id"] > 0]
#     .assign(codon=tRNA_table["codon"].str.split("/"))
#     .explode("codon")[["trna", "codon"]]
# )
def k_calculator(codon, tRNA_table, tRNA_codon_table, codon_table):
    k = 0
    for tRNA in tRNA_codon_table[tRNA_codon_table["codon"] == codon]["trna"]:
        _c = tRNA_table[tRNA_table["trna"] == tRNA]["concentration"].values[0]
        if _c == -1:
            return pd.NA
        _c *= codon_table[codon_table["r_codon"] == codon]["usage"].values[0]
        _u = 0
        for _codon in tRNA_codon_table[tRNA_codon_table["trna"] == tRNA]["codon"]:
            _u += codon_table[codon_table["r_codon"] == _codon]["usage"].values[0]
        _c /= _u
        k += _c
    return k
