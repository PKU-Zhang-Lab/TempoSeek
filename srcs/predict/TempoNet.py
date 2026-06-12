"""
TempoNet inference: PDB/CIF -> CSV with per-residue tempo prediction.

Usage:
    python -m srcs.predict.TempoNet -i ./pdbs
    python srcs/predict/TempoNet.py -i ./pdbs
"""

import argparse
import numpy as np
import sys
import time
import torch
from pathlib import Path
from tqdm import tqdm
import os

# Ensure project root is on sys.path so `srcs.*` imports work
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from utils.utils import aa_alphabet
from predict import TempoNetPredictDataLoader
from model.TempoNet import TempoNet
from predict.utils import Logger, tempo_to_b


_AA_INDEX_TO_LETTER = {i: c for i, c in enumerate(aa_alphabet)}


def _aa_idx_to_letter(idx: int) -> str:
    if 0 <= idx < 20:
        return _AA_INDEX_TO_LETTER[idx]
    return "X"


def _interpolate_1d(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if valid_mask.all():
        return arr
    orig_shape = arr.shape
    flat = arr.reshape(len(arr), -1)
    x = np.arange(len(arr))
    for c in range(flat.shape[1]):
        col = flat[:, c]
        if not np.all(np.isfinite(col[valid_mask])):
            continue
        if valid_mask.sum() >= 2:
            flat[:, c] = np.interp(x, x[valid_mask], col[valid_mask])
        else:
            flat[~valid_mask, c] = 0.0
    return flat.reshape(orig_shape)


def predict(
    files: list[tuple[str, str, str]],
    checkpoint: str,
    batch_size: int,
    device: torch.device,
    quiet: bool,
    logger: Logger,
) -> list[Path]:
    """Run TempoNet inference.

    Args:
        files: List of (input_pdb, output_csv, pkl_cache) tuples
        checkpoint: Model checkpoint path
        batch_size: Token limit per batch
        device: torch device
        quiet: Suppress all print output

    Returns:
        Generated CSV file paths
    """
    logger.log(f"Loading checkpoint: {checkpoint}", "INFO")
    try:
        model = TempoNet.load_from_checkpoint(checkpoint).to(device).eval()
    except Exception as e:
        logger.log(f"Failed to load checkpoint '{checkpoint}': {e}", "ERROR", e)
        return []

    try:
        loader = TempoNetPredictDataLoader(files=files, batch_size=batch_size, device=device, quiet=quiet)
    except Exception as e:
        logger.log(f"Failed to create DataLoader: {e}", "ERROR", e)
        return []
    logger.log(f"Loader ready: {len(loader)} batches, {len(files)} files", "INFO")

    csv_paths = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Inference", unit="batch", disable=quiet)):
            try:
                X, A, IC, IR, MV, ResNo, Len, Name, Out, ChainName = batch
                pred = model(X, A, MV, IR, IC)[0].cpu().numpy().squeeze(-1)
                Len = Len.cpu().numpy()
            except Exception as e:
                logger.log(f"Batch {batch_idx} inference failed: {e}", "WARN", e)
                continue

            for i, name in enumerate(Name):
                try:
                    out = Path(Out[i])
                    L = Len[i]
                    pred_i, aa_i, resno_i = (
                        pred[i, :L], A[i, :L].cpu().numpy(),
                        ResNo[i, :L].cpu().numpy(),
                    )
                    mask_i = MV[i, :L].cpu().numpy().astype(bool)
                    if not mask_i.all():
                        pred_i = _interpolate_1d(pred_i, mask_i)

                    chain_labels = ChainName[i][:L]
                    aa_letters = [_aa_idx_to_letter(a) for a in aa_i]
                    with open(out, "w") as f:
                        f.write("res_idx,chain,aa,tempo\n")
                        for r, c, a, p in zip(resno_i, chain_labels, aa_letters, pred_i):
                            f.write(f"{r},{c},{a},{p:.6f}\n")
                    csv_paths.append(out)
                except Exception as e:
                    logger.log(f"Failed to save '{name}': {e}", "WARN", e)
                    continue

    logger.log(f"Done! {len(csv_paths)}/{len(files)} CSV(s) generated", "INFO")
    return csv_paths


_VALID_SUFFIXES = (".pdb", ".cif", ".pdb.gz", ".cif.gz")

def preprocess(input_dir: str, output_dir: str, logger: Logger, force: bool) -> list[tuple[str, str, str]]:
    """Scan input directory for PDB/CIF files and prepare file tuples for DataLoader."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        logger.log(f"Input path '{input_dir}' is not a directory", "ERROR", ValueError(f"Input path '{input_dir}' is not a directory"))
        return [], None

    files = []
    out_dir = Path(output_dir) if output_dir else input_path / "temponet_output"
    tmp_dir = input_path / "cache"
    if tmp_dir.exists():
        if not force:
            logger.log(
                f"Temporary cache directory '{tmp_dir}' already exists. "
                f"Due to scheduled deletion after inference, this folder will be removed after inference if --reserve-cache is not set. "
                f"If you believe what you do is correct, use --force to ignore this warning and this will cause deletion of '{tmp_dir}' after inference.",
                "WARN",
                FileExistsError(
                    f"Temporary cache directory '{tmp_dir}' already exists"
                ),
            )
            return [], None
        else:
            logger.log(
                f"Temporary cache directory '{tmp_dir}' already exists. Overwriting due to --force. This will cause deletion of '{tmp_dir}' after inference if --reserve-cache is not set.",
                "WARN",
                FileExistsError(
                    f"Temporary cache directory '{tmp_dir}' already exists"
                ),
            )
    if out_dir.exists():
        if not force:
            logger.log(
                f"Output directory '{out_dir}' already exists. Please remove or use --force to overwrite.",
                "ERROR",
                FileExistsError(f"Output directory '{out_dir}' already exists"),
            )
            return [], None
        else:
            logger.log(
                f"Output directory '{out_dir}' already exists. Overwriting due to --force.",
                "WARN",
                FileExistsError(f"Output directory '{out_dir}' already exists"),
            )
    for file in input_path.iterdir():
        if any(file.name.endswith(suffix) for suffix in _VALID_SUFFIXES):
            name = file.name.removesuffix(".gz").rsplit(".", 1)[0]
            out_path = out_dir / f"{name}_tempo.csv"
            cache_path = tmp_dir / f"{name}.pkl"
            files.append((str(file), str(out_path), str(cache_path)))
    if files:
        out_dir.mkdir(exist_ok=True)
        tmp_dir.mkdir(exist_ok=True)
    else:
        logger.log(f"No valid PDB/CIF files found in '{input_dir}'", "WARN", ValueError("No input files"))
    return files, tmp_dir

def postprocess(files: list[tuple[str, str, str]], tmp_dir: Path, logger: Logger) -> list[Path]:
    """Remove temporary cache files after inference."""
    for _, _, cache_path in files:
        try:
            os.remove(cache_path)
        except OSError as e:
            logger.log(f"Failed to remove cache file '{cache_path}': {e}", "WARN", e)
    try:
        tmp_dir.rmdir()
    except OSError as e:
        logger.log(f"Failed to remove temporary directory '{tmp_dir}': {e}", "WARN", e)
    return [Path(out) for _, out, _ in files]


def save_figure(csv_paths: list[Path], logger: Logger):
    """Read tempo CSVs and save profiles as PNGs."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
            for chain in df["chain"].unique():            
                fig_path = csv_path.with_suffix(".png")
                fig_path = fig_path.with_name(fig_path.stem + (f"_chain{chain}" if len(df["chain"].unique()) > 1 else ""))
                chain_df = df[df["chain"] == chain]
                plt.figure(figsize=(12, 8))
                plt.plot(chain_df["res_idx"], chain_df["tempo"], "b-", linewidth=1)
                plt.xlabel("Residue")
                plt.ylabel("Tempo")
                plt.ylim(0, 12)
                plt.title(f"{csv_path.stem.removesuffix('_tempo')} Translation Tempo" + (f" - Chain {chain}" if len(df["chain"].unique()) > 1 else ""))
                plt.tight_layout()
                plt.savefig(fig_path, dpi=150)
                plt.close()
        except Exception as e:
            logger.log(f"Failed to save figure for '{csv_path.stem}': {e}", "WARN", e)


def save_pdb_with_bfactor(pairs: list[tuple[str, Path]], logger: Logger):
    """Save PDB copies with tempo as b-factor for each (pdb_path, csv_path) pair."""
    try:
        import gzip
        import pandas as pd
        from Bio.PDB import PDBParser, FastMMCIFParser, PDBIO
    except ImportError:
        logger.log("Bio.PDB not available, skipping b-factor PDB output", "WARN")
        return

    for pdb_path, csv_path in tqdm(pairs, desc="B-factor PDB", unit="pdb"):
        try:
            df = pd.read_csv(csv_path)
            tempo_map = {(row["chain"], row["res_idx"]): row["tempo"] for _, row in df.iterrows()}

            gz = pdb_path.endswith(".gz")
            tp = ".pdb" if ".pdb" in pdb_path else ".cif"
            f = gzip.open(pdb_path, "rt") if gz else open(pdb_path, "r")
            parser = PDBParser(QUIET=True) if tp == ".pdb" else FastMMCIFParser(QUIET=True)
            structure = parser.get_structure("x", f)
            f.close()

            for chain in structure.get_chains():
                cid = chain.get_id()
                for res in chain.get_residues():
                    key = (cid, res.get_id()[1])
                    if key in tempo_map:
                        b_val = float(tempo_to_b(tempo_map[key]))
                        for atom in res.get_atoms():
                            atom.set_bfactor(b_val)

            out_path = csv_path.with_suffix(".pdb")
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(out_path))
        except Exception as e:
            logger.log(f"Failed to save b-factor PDB for '{csv_path.stem}': {e}", "WARN", e)


def main():
    parser = argparse.ArgumentParser(description="TempoNet inference")
    parser.add_argument("-i", "--input-dir", required=True, dest="input_dir", help="Input directory containing PDB/CIF files")
    parser.add_argument("-o", "--output-dir", default=None, dest="output_dir", help="Output directory for tempo predictions")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress all output")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite existing output directory")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", help="Print full error tracebacks")
    parser.add_argument("--checkpoint", default="ckpts/temponet.ckpt")
    parser.add_argument("--device", default="cuda", help="Device for inference (e.g. 'cpu', 'cuda')")
    parser.add_argument("--reserve-cache", action="store_true", help="Keep temporary cache files for debugging")
    parser.add_argument("--figure", action="store_true", help="Save tempo profile as PNG for each protein")
    parser.add_argument("--bfactor", action="store_true", help="Save a copy of the input PDB with tempo values in the B-factor column")
    args = parser.parse_args()

    logger = Logger(quiet=args.quiet, debug=args.debug)

    try:
        device = torch.device(args.device)
    except Exception as e:
        logger.log(f"Invalid device '{args.device}': {e}", "ERROR", e)
        return
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.log("CUDA device specified but not available. Please check your PyTorch installation and GPU setup. If using CPU, specify '--device cpu'.", "ERROR", RuntimeError("CUDA not available"))
        return


    files, tmp_dir = preprocess(args.input_dir, args.output_dir, logger, force=args.force)
    if not files:
        return
    csv_paths = predict(
        files=files,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        device=device,
        quiet=args.quiet,
        logger=logger,
    )

    if args.figure:
        save_figure(csv_paths, logger)

    if args.bfactor:
        pairs = [(inp, Path(out)) for inp, out, _ in files]
        save_pdb_with_bfactor(pairs, logger)

    if not args.reserve_cache:
        postprocess(files, tmp_dir, logger)

if __name__ == "__main__":
    main()
