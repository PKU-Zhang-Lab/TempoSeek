"""
TempoCoder inference: CSV -> CSV with codon design.

Input CSV columns: residue_seq, chain, aa, tempo
Output CSV columns: residue_seq, aa, input_tempo, codon, codon_tempo

Usage:
    python -m srcs.predict.TempoCoder -i ./results -o ./codon
    python srcs/predict/TempoCoder.py -i ./results -o ./codon
"""

import argparse
import numpy as np
import sys
import torch
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from predict import TempoCoderPredictDataLoader
from model.TempoCoder import TempoCoder
from predict.utils import Logger


def predict(
    files: list[tuple[str, str]],
    checkpoint: str,
    batch_size: int,
    device: torch.device,
    temperature: float = 1.0,
    n_samples: int = 1,
    quiet: bool = False,
    logger: Logger = None,
) -> list[Path]:
    """Run TempoCoder inference.

    Args:
        files: List of (input_csv, output_csv) tuples
        checkpoint: Model checkpoint path
        batch_size: Token limit per batch
        device: torch device
        temperature: Sampling temperature for codon design
        n_samples: Number of codon samples per residue
        quiet: Suppress all print output
        logger: Logger instance

    Returns:
        Generated CSV file paths
    """
    if logger is None:
        logger = Logger(quiet=quiet)

    logger.log(f"Loading checkpoint: {checkpoint}")
    try:
        model = TempoCoder.load_from_checkpoint(checkpoint).to(device).eval()
    except Exception as e:
        logger.log(f"Failed to load checkpoint '{checkpoint}': {e}", "ERROR", e)
        return []

    try:
        loader = TempoCoderPredictDataLoader(
            files=files, batch_size=batch_size, device=device, quiet=quiet,
        )
    except Exception as e:
        logger.log(f"Failed to create DataLoader: {e}", "ERROR", e)
        return []
    logger.log(f"Loader ready: {len(loader)} batches, {len(files)} files")

    from utils.utils import aa_alphabet, codon_list, tempo_calculator

    out_paths = []
    with torch.no_grad():
        for A, T, MV, RS, names, outs, rs_valid in tqdm(
            loader, desc="Codon design", unit="batch", disable=quiet
        ):
            try:
                logits, _ = model(A, T, MV)
            except Exception as e:
                logger.log(f"Batch inference failed: {e}", "WARN", e)
                continue
            B, N = logits.shape[:2]

            for s in tqdm(range(n_samples), desc="Samples", unit="sample", disable=quiet or n_samples <= 1, leave=False):
                # Slice to 64 standard codons, drop NNN(64) & padding(65)
                logits_64 = logits[..., :64]
                if temperature == 0:
                    codons = logits_64.argmax(dim=-1)
                else:
                    probs = torch.softmax(logits_64 / temperature, dim=-1)
                    codons = torch.multinomial(
                        probs.view(-1, probs.size(-1)), num_samples=1
                    ).view(B, N)

                codons_np = codons.cpu().numpy()
                A_np = A.cpu().numpy()
                T_np = T.cpu().numpy()
                RS_np = RS.cpu().numpy()

                for b in range(B):
                    try:
                        name = names[b]
                        out_path = Path(outs[b])
                        if n_samples > 1:
                            out_path = out_path.with_stem(f"{out_path.stem}_sample_{s}")

                        # Build DNA sequence from predicted codons
                        dna_seq = ""
                        valid_indices = []
                        for i in range(N):
                            if MV[b, i].item() == 0:
                                continue
                            dna_seq += codon_list[codons_np[b, i]]
                            valid_indices.append(i)

                        # Calculate codon-level tempo
                        codon_tempo = tempo_calculator(dna_seq)

                        rows = []
                        for j, i in enumerate(valid_indices):
                            aa = aa_alphabet[A_np[b, i]] if 0 <= A_np[b, i] < 20 else "X"
                            codon = codon_list[codons_np[b, i]]
                            ct = f"{codon_tempo[j]:.6f}" if j < len(codon_tempo) else ""
                            rows.append([
                                RS_np[b, i], aa,
                                f"{T_np[b, i]:.6f}", codon, ct,
                            ])

                        with open(out_path, "w") as f:
                            f.write("res_idx,aa,input_tempo,codon,codon_tempo\n")
                            for row in rows:
                                f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")

                        # Save DNA sequence as .txt
                        txt_path = out_path.with_suffix(".txt")
                        with open(txt_path, "w") as f:
                            f.write(dna_seq + "\n")
                        out_paths.append(out_path)
                    except Exception as e:
                        logger.log(f"Failed to save '{name}': {e}", "WARN", e)
                        continue

    logger.log(f"Done! {len(out_paths)}/{len(files) * n_samples} CSV(s) generated")
    return out_paths


def preprocess(
    input_dir: str,
    output_dir: str,
    logger: Logger,
    force: bool = False,
    remove_tempo_suffix: bool = True,
) -> list[tuple[str, str]]:
    """Scan input directory for CSV files and prepare file tuples for DataLoader.

    Args:
        input_dir: Directory containing TempoNet CSV outputs
        output_dir: Output directory (auto-derived if None)
        logger: Logger instance
        force: Force overwrite existing output directory

    Returns:
        List of (input_csv, output_csv) tuples
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir) if output_dir else in_path / "tempocoder_csv"

    if not in_path.exists():
        logger.log(
            f"Input directory '{input_dir}' not found",
            "ERROR",
            FileNotFoundError(f"Input directory '{input_dir}' not found"),
        )
        return []
    
    if out_path.exists():
        if not force:
            logger.log(
                f"Output directory '{out_path}' already exists. "
                f"Use --force to overwrite.",
                "ERROR",
                FileExistsError(f"Output directory '{out_path}' already exists"),
            )
            return []
        logger.log(f"Output directory '{out_path}' already exists. Overwriting due to --force.", "WARN")

    # Collect input CSV files
    csv_files = sorted(in_path.glob("*.csv"))
    if not csv_files:
        logger.log(f"No CSV files found in '{input_dir}'", "WARN", FileNotFoundError(f"No CSV files in '{input_dir}'"))
        return []

    out_path.mkdir(parents=True, exist_ok=True)

    files = []
    for f in csv_files:
        name = f.stem
        if remove_tempo_suffix:
            name = name.removesuffix("_tempo")
        out_csv = out_path / name / f"{name}_codon.csv"
        (out_path / name).mkdir(parents=True, exist_ok=True)
        files.append((str(f), str(out_csv)))

    logger.log(f"Found {len(files)} CSV file(s) in '{input_dir}'")
    return files

def save_figure(csv_paths: list[Path], logger: Logger):
    """Read codon CSVs and save dual-panel profiles as PNGs."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return

    for csv_path in tqdm(csv_paths, desc="Figures", unit="png"):
        try:
            df = pd.read_csv(csv_path)
            fig_path = csv_path.with_suffix(".png")
            plt.figure(figsize=(8, 6))
            plt.plot(df["res_idx"], df["input_tempo"], "b-", linewidth=1, label="Input Tempo")
            plt.plot(df["res_idx"], df["codon_tempo"], "r-", linewidth=1, label="Codon Tempo")
            plt.xlabel("Residue")
            plt.ylabel("Tempo")
            plt.ylim(0, 12)
            plt.legend()
            plt.title(csv_path.stem)
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
        except Exception as e:
            logger.log(f"Failed to save figure for '{csv_path.stem}': {e}", "WARN", e)



def main():
    parser = argparse.ArgumentParser(description="TempoCoder inference")
    parser.add_argument("-i", "--input-dir", required=True, dest="input_dir",
                        help="Input directory containing CSV files")
    parser.add_argument("-o", "--output-dir", default=None, dest="output_dir",
                        help="Output directory for codon predictions")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress all output")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force overwrite existing output directory")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--debug", action="store_true", help="Print full error tracebacks")
    parser.add_argument("--checkpoint", default="ckpts/tempocoder.ckpt")
    parser.add_argument("--device", default="cuda",
                        help="Device for inference (e.g. 'cpu', 'cuda')")
    parser.add_argument("-t", "--temperature", type=float, default=0,
                        help="Sampling temperature for codon design")
    parser.add_argument("-n", "--n-samples", type=int, default=1,
                        help="Number of codon samples per residue")
    parser.add_argument("--figure", action="store_true", help="Save codon profile as PNG for each output")
    args = parser.parse_args()

    logger = Logger(quiet=args.quiet, debug=args.debug)

    if args.n_samples > 1 and args.temperature == 0:
        logger.log("Sampling multiple codon designs with temperature=0 is not useful. Setting n_samples to 1.", "WARN")
        args.n_samples = 1

    try:
        device = torch.device(args.device)
    except Exception as e:
        logger.log(f"Invalid device '{args.device}': {e}", "ERROR", e)
        return
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.log("CUDA device specified but not available. Please check your PyTorch installation and GPU setup. If using CPU, specify '--device cpu'.", "ERROR", RuntimeError("CUDA not available"))
        return

    files = preprocess(args.input_dir, args.output_dir, logger, force=args.force)
    if not files:
        return

    out_paths = predict(
        files=files,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        device=device,
        temperature=args.temperature,
        n_samples=args.n_samples,
        quiet=args.quiet,
        logger=logger,
    )

    if args.figure:
        save_figure([Path(p) for p in out_paths], logger)


if __name__ == "__main__":
    main()
