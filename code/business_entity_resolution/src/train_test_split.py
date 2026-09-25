import pandas as pd
import os
import gc

# ============================================================
# CONFIG
# ============================================================

SOURCE_DIR = r"D:\ML_HACK\data\6ab10eb3b23ba_student_resource\student_resource\dataset\train"

OUTPUT_TRAIN = r"D:\ML_HACK\data\new\train"
OUTPUT_TEST = r"D:\ML_HACK\data\new\test"

TRAIN_RATIO = 0.90
RANDOM_STATE = 42

os.makedirs(OUTPUT_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_TEST, exist_ok=True)


# ============================================================
# FILE PATHS
# ============================================================

s1_file = os.path.join(SOURCE_DIR, "train_source1.tsv")
s2_file = os.path.join(SOURCE_DIR, "train_source2.tsv")
s3_file = os.path.join(SOURCE_DIR, "train_source3.tsv")
gt_file = os.path.join(SOURCE_DIR, "train_ground_truth.tsv")


# ============================================================
# STEP 1 — READ GROUND TRUTH
# ============================================================

print("=" * 60)
print("STEP 1: Reading ground truth...")
print("=" * 60)

gt = pd.read_csv(
    gt_file,
    sep="\t",
    dtype=str
)

print(f"Ground truth rows: {len(gt):,}")


# ============================================================
# STEP 2 — RANDOMLY SELECT 10% OF S1 FOR VALIDATION
# ============================================================

print("\nSelecting validation S1 records...")

all_s1_ids = gt["source1_entity_id"].drop_duplicates()

test_s1_ids = set(
    all_s1_ids.sample(
        frac=1 - TRAIN_RATIO,
        random_state=RANDOM_STATE
    )
)

train_s1_ids = set(all_s1_ids) - test_s1_ids

print(f"Total S1 IDs : {len(all_s1_ids):,}")
print(f"Train S1 IDs : {len(train_s1_ids):,}")
print(f"Test S1 IDs  : {len(test_s1_ids):,}")


# ============================================================
# STEP 3 — SPLIT GROUND TRUTH
# ============================================================

print("\nExtracting S2/S3 IDs for validation...")

gt_test = gt[
    gt["source1_entity_id"].isin(test_s1_ids)
]

gt_train = gt[
    gt["source1_entity_id"].isin(train_s1_ids)
].copy()


# ============================================================
# STEP 4 — EXTRACT ALL MATCHED IDs BELONGING TO TEST S1s
# ============================================================

test_matched_ids = set()

for value in gt_test["matched_entity_ids"].dropna():

    if not value:
        continue

    for entity_id in value.split(","):

        entity_id = entity_id.strip()

        if entity_id:
            test_matched_ids.add(entity_id)


print(f"Test S2/S3 IDs: {len(test_matched_ids):,}")


# ============================================================
# SAVE GROUND TRUTH SPLITS
# ============================================================

print("\nSaving ground truth splits...")

gt_train.to_csv(
    os.path.join(
        OUTPUT_TRAIN,
        "train_ground_truth.tsv"
    ),
    sep="\t",
    index=False
)

gt_test.to_csv(
    os.path.join(
        OUTPUT_TEST,
        "test_ground_truth.tsv"
    ),
    sep="\t",
    index=False
)


# ============================================================
# DELETE GROUND TRUTH FROM RAM
# ============================================================

del gt
del gt_test
del gt_train
del all_s1_ids

gc.collect()

print("Ground truth removed from RAM.")


# ============================================================
# STEP 5 — PROCESS S1
# ============================================================

print("\n" + "=" * 60)
print("STEP 2: Processing S1...")
print("=" * 60)

s1 = pd.read_csv(
    s1_file,
    sep="\t",
    dtype=str
)

print(f"S1 loaded: {len(s1):,} rows")


s1_train = s1[
    s1["entity_id"].isin(train_s1_ids)
]

s1_test = s1[
    s1["entity_id"].isin(test_s1_ids)
]


s1_train.to_csv(
    os.path.join(
        OUTPUT_TRAIN,
        "train_source1.tsv"
    ),
    sep="\t",
    index=False
)

s1_test.to_csv(
    os.path.join(
        OUTPUT_TEST,
        "test_source1.tsv"
    ),
    sep="\t",
    index=False
)


print(f"S1 train: {len(s1_train):,}")
print(f"S1 test : {len(s1_test):,}")


# Delete S1 DataFrames

del s1
del s1_train
del s1_test

gc.collect()

print("S1 removed from RAM.")


# ============================================================
# STEP 6 — PROCESS S2
# ============================================================

print("\n" + "=" * 60)
print("STEP 3: Processing S2...")
print("=" * 60)

s2 = pd.read_csv(
    s2_file,
    sep="\t",
    dtype=str
)

print(f"S2 loaded: {len(s2):,} rows")


s2_test = s2[
    s2["entity_id"].isin(test_matched_ids)
]

s2_train = s2[
    ~s2["entity_id"].isin(test_matched_ids)
]


s2_train.to_csv(
    os.path.join(
        OUTPUT_TRAIN,
        "train_source2.tsv"
    ),
    sep="\t",
    index=False
)

s2_test.to_csv(
    os.path.join(
        OUTPUT_TEST,
        "test_source2.tsv"
    ),
    sep="\t",
    index=False
)


print(f"S2 train: {len(s2_train):,}")
print(f"S2 test : {len(s2_test):,}")


# Delete S2 DataFrames

del s2
del s2_train
del s2_test

gc.collect()

print("S2 removed from RAM.")


# ============================================================
# STEP 7 — PROCESS S3
# ============================================================

print("\n" + "=" * 60)
print("STEP 4: Processing S3...")
print("=" * 60)

s3 = pd.read_csv(
    s3_file,
    sep="\t",
    dtype=str
)

print(f"S3 loaded: {len(s3):,} rows")


s3_test = s3[
    s3["entity_id"].isin(test_matched_ids)
]

s3_train = s3[
    ~s3["entity_id"].isin(test_matched_ids)
]


s3_train.to_csv(
    os.path.join(
        OUTPUT_TRAIN,
        "train_source3.tsv"
    ),
    sep="\t",
    index=False
)

s3_test.to_csv(
    os.path.join(
        OUTPUT_TEST,
        "test_source3.tsv"
    ),
    sep="\t",
    index=False
)


print(f"S3 train: {len(s3_train):,}")
print(f"S3 test : {len(s3_test):,}")


# Delete S3 DataFrames

del s3
del s3_train
del s3_test

gc.collect()

print("S3 removed from RAM.")


# ============================================================
# CLEAN UP SMALL LOOKUP SETS
# ============================================================

del train_s1_ids
del test_s1_ids
del test_matched_ids

gc.collect()


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("DONE!")
print("=" * 60)

print(f"\nTraining files saved to:")
print(OUTPUT_TRAIN)

print(f"\nValidation files saved to:")
print(OUTPUT_TEST)