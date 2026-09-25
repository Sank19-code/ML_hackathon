import polars as pl
gt=pl.read_parquet("work/train_gt.parquet")
s1=pl.read_parquet("work/train_s1.parquet")
s2=pl.read_parquet("work/train_s2.parquet"); s3=pl.read_parquet("work/train_s3.parquet")
gt=gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element()!="")).alias("m"))
gt=gt.with_columns(pl.col("m").list.len().alias("n"))
print("singleton frac", (gt['n']==0).mean())
print(gt['n'].value_counts().sort('n').head(20))
pairs=gt.select("source1_entity_id","m").explode("m").drop_nulls()
print("pairs",pairs.shape)
dup=pairs.group_by("m").len().filter(pl.col("len")>1)
print("S2/S3 ids matching >1 S1:", dup.shape)
pairs=pairs.with_columns(pl.col("m").str.slice(0,2).alias("src"))
print(pairs['src'].value_counts())
# per s1: count of s2 and s3
per=pairs.group_by("source1_entity_id").agg((pl.col("src")=="S2").sum().alias("n2"),(pl.col("src")=="S3").sum().alias("n3"))
print(per.group_by("n2","n3").len().sort("len",descending=True).head(20))
# fraction of s2/s3 records that are matched
matched=set(pairs['m'].to_list())
print("S2 matched frac", s2['entity_id'].is_in(matched).mean(), "S3 matched frac", s3['entity_id'].is_in(matched).mean())
# country crossing
c1=s1.select(pl.col("entity_id").alias("source1_entity_id"),pl.col("country").alias("c1"))
c23=pl.concat([s2,s3]).select(pl.col("entity_id").alias("m"),pl.col("country").alias("c2"))
j=pairs.join(c1,on="source1_entity_id").join(c23,on="m")
print("cross-country", (j['c1']!=j['c2']).sum(), "of", len(j))
# singletons by country
g=gt.join(c1,on="source1_entity_id")
print(g.group_by("c1").agg((pl.col("n")==0).mean().alias("single"),pl.col("n").mean().alias("mean_n"),pl.len()))
