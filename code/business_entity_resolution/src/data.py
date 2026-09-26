"""
Loading and normalising the raw source files.

Every source TSV is read once (all columns as strings, no quote handling), normalised
in parallel with multiprocessing, and cached as parquet in the work directory. The
cached frame keeps the raw columns plus the canonical fields produced by
normalize.normalize_name / normalize.normalize_address.
"""
import os
import time
from multiprocessing import Pool
from typing import Optional

import polars as pl

try:
    from .normalize import ADDR_FIELDS, NAME_FIELDS, normalize_batch
except ImportError:
    from normalize import ADDR_FIELDS, NAME_FIELDS, normalize_batch

INT_FIELDS = {"is_domain", "is_native", "has_alias", "a_empty"}


def read_tsv(path: str) -> pl.DataFrame:
    """Read a challenge TSV: tab separated, no quoting, every column a string."""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema_length=0,
                     null_values=[""], encoding="utf8-lossy")
    return df.with_columns(pl.col(c).fill_null("") for c in df.columns)


def read_ground_truth(path: str) -> pl.DataFrame:
    """Return long format (source1_entity_id, entity_id) plus the list of all S1 ids."""
    gt = read_tsv(path)
    return gt


def _chunks(n: int, size: int):
    for start in range(0, n, size):
        yield start, min(start + size, n)


def normalize_frame(df: pl.DataFrame, translit_path: Optional[str], n_jobs: int,
                    chunk: int = 20000) -> pl.DataFrame:
    names = df["business_name"].to_list()
    addrs = df["business_address"].to_list()
    countries = df["country"].to_list()
    tasks = [(names[a:b], addrs[a:b], countries[a:b], translit_path) for a, b in _chunks(len(df), chunk)]
    del names, addrs, countries
    parts = []
    with Pool(n_jobs) as pool:
        for out_n, out_a in pool.imap(normalize_batch, tasks, chunksize=1):
            cols = {}
            for i, f in enumerate(NAME_FIELDS):
                cols[f] = [r[i] for r in out_n]
            for i, f in enumerate(ADDR_FIELDS):
                cols[f] = [r[i] for r in out_a]
            part = pl.DataFrame(cols)
            part = part.with_columns([pl.col(c).cast(pl.Int8) for c in INT_FIELDS])
            parts.append(part)
    norm = pl.concat(parts)
    return pl.concat([df, norm], how="horizontal")


def prepare_source(tsv_path: str, out_path: str, translit_path: Optional[str], n_jobs: int,
                   force: bool = False) -> str:
    if os.path.isfile(out_path) and not force:
        return out_path
    t0 = time.time()
    df = read_tsv(tsv_path)
    df = normalize_frame(df, translit_path, n_jobs)
    df.write_parquet(out_path)
    print(f"  normalised {os.path.basename(tsv_path)}: {len(df):,} rows in {time.time()-t0:.0f}s")
    return out_path
