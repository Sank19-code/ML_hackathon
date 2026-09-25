import polars as pl, random, sys
gt=pl.read_parquet("work/train_gt.parquet")
s1=pl.read_parquet("work/train_s1.parquet")
s23=pl.concat([pl.read_parquet("work/train_s2.parquet"),pl.read_parquet("work/train_s3.parquet")])
country=sys.argv[1]; n=int(sys.argv[2]); seed=int(sys.argv[3])
s1c=s1.filter(pl.col("country")==country).sample(n,seed=seed)
g=gt.filter(pl.col("source1_entity_id").is_in(s1c['entity_id'].implode()))
d23={r[0]:(r[1],r[2]) for r in s23.filter(pl.col("entity_id").is_in(pl.Series([x for m in g['matched_entity_ids'].fill_null("").to_list() for x in m.split(",") if x]).implode())).iter_rows()}
gd=dict(g.iter_rows())
for eid,name,addr,c in s1c.iter_rows():
    print(f"### {eid} | {name} | {addr}")
    for m in (gd.get(eid) or "").split(","):
        if m: print(f"    {m[:2]} | {d23[m][0]} | {d23[m][1]}")
