"""sample.py - real lines from the 12:00 file, for the screen, and the bucket ChatGPT hashes to.
    python sample.py --data <folder> --out out"""
import os, gzip, json, zlib, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="out")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
want = {b"en Jailer_2", b"en.m Jailer_2", b"en ChatGPT", b"en.m ChatGPT"}
got, head = {}, []
with gzip.open(os.path.join(a.data, "pageviews-20261020-120000.gz"), "rb") as f:
    for line in f:
        line = line.rstrip(b"\n")
        p = line.split(b" ")
        if len(head) < 400 and p[0] == b"en" and len(p) == 4 and p[1][:1].isalpha() and int(p[2]) > 20:
            head.append(line.decode("utf-8", "replace"))
        k = b" ".join(p[:2])
        if k in want:
            got[k.decode()] = line.decode("utf-8", "replace")
out = {"hour": "12", "lines": got, "plain_en_lines": head[:400:50],
       "chatgpt_bucket": zlib.crc32(b"ChatGPT") % 8, "kim_bucket": zlib.crc32(b"Kim_Jong_Un") % 8,
       "jailer_bucket": zlib.crc32(b"Jailer_2") % 8}
json.dump(out, open(os.path.join(a.out, "sample_lines.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
