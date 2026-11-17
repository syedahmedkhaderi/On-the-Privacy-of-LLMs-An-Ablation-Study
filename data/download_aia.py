"""Download an author-approved AIA artifact into a private local workspace, verifying its hash.
No text or model answers are redistributed here. Supply the original artifact URL and its recorded SHA-256.
"""
import argparse,hashlib,pathlib,urllib.request
p=argparse.ArgumentParser();p.add_argument('--url',default='https://raw.githubusercontent.com/eth-sri/llmprivacy/d9cd9191ef927c66c41cbd5debe749e44292e3d1/data/synthetic/synthetic_dataset.jsonl');p.add_argument('--sha256',default='3c3b34f979afa73e6dec6b56ac84edd9eb8f0d93c24ed7b34bbbab005c476861');p.add_argument('--out',required=True);a=p.parse_args()
out=pathlib.Path(a.out)
if out.exists():raise SystemExit('Refusing to replace existing data')
content=urllib.request.urlopen(a.url,timeout=60).read()
if hashlib.sha256(content).hexdigest()!=a.sha256:raise SystemExit('SHA-256 mismatch')
out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(content);print('Verified download complete')
