#!/usr/bin/env python3
"""
Held-out evaluation on the labelled training universe.

`run.py --mode train` writes out-of-fold stage-2 probabilities for every training
candidate pair (models are cross-fitted by Source 1 entity, so no entity is ever scored
by a model that saw it). This script re-runs the decision tuning / evaluation from those
saved scores without retraining:

  * decision rules are tuned on one half of the Source 1 entities and scored on the other
    half (and vice versa), so tuning and evaluation never share data
  * prints macro F0.5 overall, by country, for singletons / non-singletons, pair
    precision / recall and the blocking recall ceiling
  * rewrites <model-dir>/thresholds.json with the rule tuned on all training entities

Usage:
  python code/business_entity_resolution/src/evaluate_holdout.py \
      --train-dir $DATA/dataset/train --work-dir work_retrain --model-dir model_retrained
  (--model-dir is required: point it at a retrained model folder, never at the shipped model/)
"""
import argparse
import json
import os
import sys

import polars as pl

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-dir", default="dataset/train")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--model-dir", required=True, help="retrained model folder (its thresholds.json is rewritten)")
    args = ap.parse_args()

    ws = pipeline.Workspace(args.work_dir, os.path.abspath(args.model_dir))
    gt = pipeline.gt_long(os.path.join(args.train_dir, "train_ground_truth.tsv"))
    tables = {}
    for c in pipeline.countries(ws, "train"):
        path = ws.scores("train", c)
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path}: run `run.py --mode train` first")
        tables[c] = pl.read_parquet(path)
    summary = pipeline.evaluate_and_tune(ws, tables, gt, pipeline.DEFAULT_CONFIG)
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
