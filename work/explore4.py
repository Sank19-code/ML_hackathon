import polars as pl, re, unicodedata
from collections import Counter
gt=pl.read_parquet("work/train_gt.parquet")
s1=pl.read_parquet("work/train_s1.parquet")
s2=pl.read_parquet("work/train_s2.parquet"); s3=pl.read_parquet("work/train_s3.parquet")
matched=set(x for m in gt['matched_entity_ids'].drop_nulls().to_list() for x in m.split(","))
# duplicate S1 names
print("S1 exact dup names:", s1.group_by("business_name","country").len().filter(pl.col("len")>1).shape, s1.group_by(pl.col("business_name").str.to_lowercase()).len().filter(pl.col("len")>1)['len'].sum())
print(s1.group_by("business_name").len().sort("len",descending=True).head(15))
# unmatched s23 samples
for nm,df in [("S2",s2),("S3",s3)]:
    um=df.filter(~pl.col("entity_id").is_in(list(matched)))
    print(nm,"unmatched",len(um))
    for r in um.sample(15,seed=5).iter_rows(): print("   ",r)
# scripts
def script(s):
    c=Counter()
    for ch in s:
        if ord(ch)>127 and ch.isalpha():
            try: c[unicodedata.name(ch).split()[0]]+=1
            except: pass
    return c.most_common(1)[0][0] if c else "LATIN"
for split in ["train","test"]:
  for s in [1,2,3]:
    df=pl.read_parquet(f"work/{split}_s{s}.parquet").sample(200000,seed=1)
    cnt=Counter((c,script(n)) for n,c in zip(df['business_name'].to_list(),df['country'].to_list()))
    print(split,s,{k:round(v/2000,2) for k,v in cnt.most_common(12)})
