import polars as pl
s1=pl.read_parquet("work/test_s1.parquet").filter(pl.col("country")=="France")
s2=pl.read_parquet("work/test_s2.parquet").filter(pl.col("country")=="France")
s3=pl.read_parquet("work/test_s3.parquet").filter(pl.col("country")=="France")
for nm,df in [("S1",s1),("S2",s2),("S3",s3)]:
    print("=====",nm)
    for r in df.sample(12,seed=2).iter_rows(): print("  ",r[1]," | ",r[2])
# find matches for a few S1 by shared street words
import re
s23=pl.concat([s2,s3])
for eid,name,addr,c in s1.sample(6,seed=11).iter_rows():
    toks=[t for t in re.findall(r"[a-zà-ÿ]{5,}",name.lower())]
    print("###",name,"|",addr)
    if not toks: continue
    t=max(toks,key=len)
    hits=s23.filter(pl.col("business_name").str.to_lowercase().str.contains(t,literal=True))
    num=re.findall(r"\d+",addr)
    if num: hits=hits.filter(pl.col("business_address").fill_null("").str.contains(num[0],literal=True))
    for r in hits.head(8).iter_rows(): print("     ",r[0][:2],r[1]," | ",r[2])
