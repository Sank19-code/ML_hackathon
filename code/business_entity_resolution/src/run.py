#!/usr/bin/env python3
"""
Business Entity Resolution - command line entry point.

  python code/business_entity_resolution/src/run.py --mode all \
      --train-dir dataset/train --test-dir dataset/test --output-dir output

Modes
  all       translit -> normalize -> block -> train -> predict -> validate
  prepare   translit + normalize + block for train and test (features are computed on the fly)
  train     stage 1 / stage 2 models + threshold tuning (needs `prepare`)
  predict   score the test set and write output/*.tsv (needs `train`)
  validate  run the official validator on output/
"""
import argparse
import json
import os
import subprocess
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

import pipeline  # noqa: E402


def run_validator(output_dir: str, test_dir: str, validator: str = None) -> int:
    candidates = [validator] if validator else []
    candidates += [
        os.path.join(os.path.dirname(os.path.abspath(test_dir.rstrip("/\\"))), "..", "utils", "validate_submission.py"),
        os.path.join(_here, "..", "..", "..", "data", "6ab10eb3b23ba_student_resource", "student_resource",
                     "utils", "validate_submission.py"),
    ]
    path = next((os.path.abspath(p) for p in candidates if p and os.path.isfile(p)), None)
    if path is None:
        print("validate_submission.py not found - skipping validation (pass --validator)")
        return 0
    print(f"\n-- official validator: {path}")
    return subprocess.run([sys.executable, path,
                           "--matching", os.path.join(output_dir, "matching_results.tsv"),
                           "--candidate", os.path.join(output_dir, "candidate_pairs.tsv"),
                           "--test-dir", test_dir], check=False).returncode


def main():
    ap = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    ap.add_argument("--mode", choices=["all", "prepare", "train", "predict", "validate"], default="all")
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--test-dir", default="dataset/test")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--work-dir", default="work", help="cache for normalised data, candidates and scores")
    ap.add_argument("--model-dir", default=os.path.join(_here, "..", "model"),
                    help="where models, thresholds and the transliteration dictionary are stored")
    ap.add_argument("--validator", default=None, help="path to utils/validate_submission.py")
    ap.add_argument("--n-jobs", type=int, default=pipeline.DEFAULT_CONFIG["n_jobs"])
<<<<<<< HEAD
    ap.add_argument("--reuse-stage1", action="store_true",
                    help="train / predict: keep the stage-1 model and reuse saved stage-1 scores of unchanged candidates")
    ap.add_argument("--force", action="store_true", help="recompute cached blocking")
    ap.add_argument("--train-countries", nargs="*", default=None,
                    help="train on these countries only (default: all countries with labels)")
    ap.add_argument("--predict-countries", nargs="*", default=None,
                    help="predict these test countries only (default: all)")
    ap.add_argument("--block-countries", nargs="*", default=None,
                    help="prepare: (re-)block only these countries, keep the other cached candidates")
    ap.add_argument("--stage1-sample", type=float, default=None, help="fraction of entities for stage 1")
    ap.add_argument("--stage2-sample", type=float, default=None, help="fraction of entities for stage 2")
    args = ap.parse_args()

    cfg = dict(pipeline.DEFAULT_CONFIG, n_jobs=args.n_jobs)
    cfg["train_countries"], cfg["predict_countries"] = args.train_countries, args.predict_countries
    cfg["block_countries"], cfg["reuse_stage1"] = args.block_countries, args.reuse_stage1
    for key in ("stage1_sample", "stage2_sample"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
=======
    ap.add_argument("--reuse-stage1", action="store_true", help="train: keep the stage-1 model and p1")
    ap.add_argument("--force", action="store_true", help="recompute cached blocking")
    args = ap.parse_args()

    cfg = dict(pipeline.DEFAULT_CONFIG, n_jobs=args.n_jobs)
>>>>>>> c74d74966aa5790f9e27f6c02d6e31673ab29d10
    ws = pipeline.Workspace(args.work_dir, os.path.abspath(args.model_dir))
    t0 = time.time()

    if args.mode in ("all", "prepare"):
        pipeline.stage_translit(ws, args.train_dir)
        for split, d in (("train", args.train_dir), ("test", args.test_dir)):
            pipeline.stage_normalize(ws, split, d, cfg["n_jobs"])
            pipeline.stage_block(ws, split, cfg, force=args.force)
    if args.mode in ("all", "train"):
        summary = pipeline.stage_train(ws, args.train_dir, cfg, reuse_stage1=args.reuse_stage1)
        with open(os.path.join(ws.model, "train_summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
    if args.mode in ("all", "predict"):
        pipeline.stage_predict(ws, args.output_dir, cfg)
    if args.mode in ("all", "predict", "validate"):
        rc = run_validator(args.output_dir, args.test_dir, args.validator)
        print("validator:", "PASS" if rc == 0 else f"FAIL (exit {rc})")
    print(f"total time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
