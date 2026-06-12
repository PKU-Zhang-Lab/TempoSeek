"""
TempoSeek pipeline: PDB/CIF -> TempoNet -> TempoCoder -> codon design.

Usage:
    python -m srcs.predict.TempoSeek -i ./pdbs -o ./results
    python srcs/predict/TempoSeek.py -i ./pdbs -o ./results
"""

import argparse
import sys
import torch
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from predict import predict_temponet, predict_tempocoder
from predict.utils import Logger
from predict.TempoNet import (
    save_figure as save_tn_figure,
    save_pdb_with_bfactor,
    preprocess as preprocess_temponet,
    postprocess as postprocess_temponet,
)
from predict.TempoCoder import (
    save_figure as save_tc_figure,
    preprocess as preprocess_tempocoder,
)

_VALID_SUFFIXES = (".pdb", ".cif", ".pdb.gz", ".cif.gz")


def main():
    parser = argparse.ArgumentParser(description="TempoSeek pipeline")
    parser.add_argument("-i", "--input-dir", required=True, dest="input_dir",
                        help="Input directory containing PDB/CIF files")
    parser.add_argument("-o", "--output-dir", default=None, dest="output_dir",
                        help="Output directory for all results")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress all output")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force overwrite existing output directory")
    parser.add_argument("--debug", action="store_true", help="Print full error tracebacks")
    parser.add_argument("--temponet-ckpt", default="ckpts/TempoNet.ckpt",
                        help="TempoNet checkpoint path")
    parser.add_argument("--tempocoder-ckpt", default="ckpts/TempoCoder.ckpt",
                        help="TempoCoder checkpoint path")
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--device", default="cuda",
                        help="Device for inference (e.g. 'cpu', 'cuda')")
    parser.add_argument("-t", "--temperature", type=float, default=0,
                        help="Sampling temperature for codon design")
    parser.add_argument("-n", "--n-samples", type=int, default=1,
                        help="Number of codon samples per residue")
    parser.add_argument("--figure", action="store_true",
                        help="Save profile figures (TempoNet + TempoCoder)")
    parser.add_argument("--bfactor", action="store_true",
                        help="Save PDB copy with TempoNet tempo as B-factor")
    args = parser.parse_args()

    logger = Logger(quiet=args.quiet, debug=args.debug)

    if args.n_samples > 1 and args.temperature == 0:
        logger.log("n_samples > 1 with temperature=0 is not useful. Setting n_samples to 1.", "WARN")
        args.n_samples = 1

    try:
        device = torch.device(args.device)
    except Exception as e:
        logger.log(f"Invalid device '{args.device}': {e}", "ERROR", e)
        return
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.log("CUDA not available. Use --device cpu.", "ERROR",
                   RuntimeError("CUDA not available"))
        return

    out_path = Path(args.output_dir) if args.output_dir else Path(args.input_dir) / "temposeek_output"
    tn_files, tmp_dir = preprocess_temponet(args.input_dir, str(out_path), logger, force=args.force)
    if not tn_files:
        return
    # Create output dir and cache (TempoNet's preprocess uses input_path/cache instead)
    out_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: TempoNet — structure -> tempo prediction
    # ------------------------------------------------------------------
    logger.log("=" * 50)
    logger.log("Step 1/2: TempoNet")
    logger.log("=" * 50)

    tn_csvs = predict_temponet(
        files=tn_files,
        checkpoint=args.temponet_ckpt,
        batch_size=args.batch_size,
        device=device,
        quiet=args.quiet,
        logger=logger,
    )

    if not tn_csvs:
        logger.log("TempoNet returned no results. Aborting.", "ERROR")
        return

    # TempoNet post-processing
    if args.figure:
        save_tn_figure(tn_csvs, logger)

    if args.bfactor:
        pairs = [(inp, Path(out)) for inp, out, _ in tn_files]
        save_pdb_with_bfactor(pairs, logger)

    # Cleanup cache
    postprocess_temponet(tn_files, tmp_dir, logger)

    # ------------------------------------------------------------------
    # Step 2: TempoCoder — tempo -> codon design
    # ------------------------------------------------------------------
    logger.log("=" * 50)
    logger.log("Step 2/2: TempoCoder")
    logger.log("=" * 50)

    tc_inputs = preprocess_tempocoder(str(out_path), str(out_path), logger, force=True, remove_tempo_suffix=True)

    if not tc_inputs:
        logger.log("No TempoCoder inputs. Skipping.", "WARN")
        tc_csvs = []
    else:
        logger.log(f"Running TempoCoder on {len(tc_inputs)} input(s)")
        tc_csvs = predict_tempocoder(
            files=tc_inputs,
            checkpoint=args.tempocoder_ckpt,
            batch_size=args.batch_size,
            device=device,
            temperature=args.temperature,
            n_samples=args.n_samples,
            quiet=args.quiet,
            logger=logger,
        )

    # TempoCoder figures
    if args.figure and tc_csvs:
        save_tc_figure([Path(p) for p in tc_csvs], logger)

    logger.log("=" * 50)
    logger.log(f"TempoSeek complete! Results in: {out_path}")
    logger.log("=" * 50)


if __name__ == "__main__":
    main()
