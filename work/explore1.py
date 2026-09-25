import polars as pl, time
D="data/6ab10eb3b23ba_student_resource/student_resource/dataset/"
t=time.time()
def load(p): return pl.read_csv(D+p, separator="\t", quote_char=None, infer_schema_length=0, null_values=[""])
for split in ["train","test"]:
    for s in [1,2,3]:
        df=load(f"{split}/{split}_source{s}.tsv")
        df.write_parquet(f"work/{split}_s{s}.parquet")
        print(split,s,df.shape, df['country'].value_counts().sort('count',descending=True).to_dicts(), "null names",df['business_name'].null_count(),"null addr",df['business_address'].null_count())
gt=load("train/train_ground_truth.tsv"); gt.write_parquet("work/train_gt.parquet")
print(gt.shape, time.time()-t)
