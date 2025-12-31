import pyarrow.parquet as pq

def parse_imagenet_path(path: str):
    if "/" in path:
        wnid, filename = path.split("/", 1)
        return wnid, filename

    parts = path.split("_")
    if len(parts) >= 3 and parts[0].startswith("n"):
        wnid = parts[0]
        return wnid, path

    raise ValueError(path)


pq_file = "/mnt/sda/weizixiang/wk/data/imagenet_v2/datasets--imagenet-1k/data/train-00000-of-00294.parquet"
table = pq.read_table(pq_file)
row = table.to_pylist()[0]

wnid, filename = parse_imagenet_path(row["image"]["path"])

print("label:", row["label"])
print("image.path:", row["image"]["path"])
print("wnid:", wnid)
print("filename:", filename)
print("bytes:", len(row["image"]["bytes"]))
