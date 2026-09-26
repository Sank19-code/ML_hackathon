#!/usr/bin/env python3
"""
Business Entity Resolution – CLI entry point.

Usage:
  # Full run (train + predict):
  python3 src/run.py --mode all

  # Train only:
  python3 src/run.py --mode train

  # Predict only (needs existing model.pkl):
  python3 src/run.py --mode predict
"""

import argparse
import os
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Make src2 importable whether run as a script or as a module
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

_project_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
_default_train_dir = os.path.join(_project_root, "data", "new", "train")
_default_test_dir = os.path.join(_project_root, "data", "new", "test")
_default_output_dir = os.path.join(_project_root, "output", "local")
_default_model_path = os.path.join(_here, "models", "model2.pkl")

from pipeline import run_train, run_predict          # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--mode", choices=["all", "train", "predict"], default="all",
                        help="Pipeline mode (default: all)")
    parser.add_argument("--train-dir", default=_default_train_dir,
                        help="Directory with train_source*.tsv + train_ground_truth.tsv")
    parser.add_argument("--test-dir", default=_default_test_dir,
                        help="Directory with test_source*.tsv")
    parser.add_argument("--output-dir", default=_default_output_dir,
                        help="Output directory for TSV files")
    parser.add_argument("--model-path", default=_default_model_path,
                        help="Path to save / load the trained model")
    parser.add_argument("--max-candidates", type=int, default=30,
                        help="Max blocking candidates per S1 entity (default: 30)")
    parser.add_argument("--train-size", type=int, default=400000,
                        help="Number of S1 entities used for training (default: 400000)")
    parser.add_argument("--val-size", type=int, default=50000,
                        help="Number of S1 entities held out for validation (default: 50000)")
    args = parser.parse_args()

    if args.mode in ("all", "train"):
        run_train(
            train_dir=args.train_dir,
            model_save_path=args.model_path,
            train_sample_size=args.train_size,
            val_sample_size=args.val_size,
            max_candidates=args.max_candidates,
        )

    if args.mode in ("all", "predict"):
        run_predict(
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            max_candidates=args.max_candidates,
        )

        # Auto-validate submission
        validator = os.path.abspath(
            os.path.join(
                _here,
                "..",
                "..",
                "..",
                "data",
                "6ab10eb3b23ba_student_resource",
                "student_resource",
                "utils",
                "validate_submission.py",
            )
        )
        if os.path.isfile(validator):
            print("\n-- Running official submission validator --")
            result = subprocess.run(
                [
                    sys.executable,
                    validator,
                    "--matching",
                    os.path.join(args.output_dir, "matching_results.tsv"),
                    "--candidate",
                    os.path.join(args.output_dir, "candidate_pairs.tsv"),
                    "--test-dir",
                    args.test_dir,
                ],
                check=False,
            )
            if result.returncode == 0:
                print("[OK] Submission files PASSED all checks - safe to upload!")
            else:
                print("[WARN] Validator reported issues. Fix before submitting.")


if __name__ == "__main__":
    main()
