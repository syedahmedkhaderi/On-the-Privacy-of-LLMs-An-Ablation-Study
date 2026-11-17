import os,json,time,datetime,socket,subprocess,hashlib
from pathlib import Path
B=Path('work/cyclea-h16');P=Path('work/cyclea-protected/h16');H=Path('work/cyclea-h12')
os.umask(0o077)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stamp():return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def log(s):
 with (B/'logs/H16_LOG.md').open('a') as f:f.write(utc()+' '+s+'\n')
def netenv():
 e=dict(os.environ);e.pop('LD_LIBRARY_PATH',None);return e
def receipt(seg,event,rid,**kw):
 r=dict(utc=utc(),host=socket.gethostname(),gpu_uuid=os.environ.get('GPU_UUID'),run_id=rid,event=event,**kw)
 with (B/'lanes'/(seg+'.receipts.jsonl')).open('a') as f:f.write(json.dumps(r)+'\n')
def wait_gpu():
 g=os.environ['GPU_UUID']
 while True:
  apps=subprocess.check_output(['nvidia-smi','-i',g,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  free=int(subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
  if not apps and free>=21000:return
  time.sleep(10)
def run(cmd,path):
 wait_gpu()
 with Path(path).open('x') as f:return subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
