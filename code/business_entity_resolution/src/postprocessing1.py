import pandas as pd

input_file = r"D:\ML_HACK\output\matching_results.tsv"
output_file = r"D:\ML_HACK\output\processed.tsv"

# Read TSV
df = pd.read_csv(input_file, sep="\t", dtype=str)

# Make sure empty cells are treated as empty strings
df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")

# ---------------------------------------------------------
# 1. Count every matched ID globally
# ---------------------------------------------------------

id_counts = {}

for value in df["matched_entity_ids"]:
    if not value:
        continue

    ids = [x.strip() for x in value.split(",") if x.strip()]

    for entity_id in ids:
        id_counts[entity_id] = id_counts.get(entity_id, 0) + 1


# ---------------------------------------------------------
# 2. IDs that occur more than once
# ---------------------------------------------------------

duplicate_ids = {
    entity_id
    for entity_id, count in id_counts.items()
    if count > 1
}

print(f"Total unique matched IDs: {len(id_counts):,}")
print(f"Duplicate IDs: {len(duplicate_ids):,}")


# ---------------------------------------------------------
# 3. Remove duplicate IDs from EVERY occurrence
# ---------------------------------------------------------

def remove_duplicates(value):
    if not value:
        return ""

    ids = [x.strip() for x in value.split(",") if x.strip()]

    # Keep only IDs that occur exactly once globally
    ids = [entity_id for entity_id in ids if entity_id not in duplicate_ids]

    return ",".join(ids)


df["matched_entity_ids"] = df["matched_entity_ids"].apply(remove_duplicates)


# ---------------------------------------------------------
# 4. Save
# ---------------------------------------------------------

df.to_csv(output_file, sep="\t", index=False)

print(f"Saved cleaned file to: {output_file}")