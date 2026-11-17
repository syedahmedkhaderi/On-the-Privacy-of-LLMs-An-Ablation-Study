import os,json,time,datetime,socket,subprocess,hashlib,fcntl
from pathlib import Path
B=Path('work/cyclea-h18');P=Path('work/cyclea-protected/h18');H=Path('work/cyclea-h12')
os.umask(0o077)
for d in ['ba','runs','dea_tokens','aia/raw','aia/jobs','aia/logs','aia/pilot','aia/scored','logs','models']:(P/d).mkdir(parents=True,exist_ok=True,mode=0o700)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def stamp():return datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8*1024*1024),b''):h.update(c)
 return h.hexdigest()
def log(s):
 with (B/'logs/H18_LOG.md').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(utc()+' '+s+'\n')
def netenv():
 e=dict(os.environ);e.pop('LD_LIBRARY_PATH',None);return e
def append(p,r):
 with Path(p).open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(r)+'\n');f.flush();os.fsync(f.fileno())
def receipt(block,event,rid,**kw):append(B/'queues'/(block+'.receipts.jsonl'),dict(utc=utc(),host=socket.gethostname(),gpu_uuid=os.environ.get('GPU_UUID'),run_id=rid,block=block,job_key=os.environ.get('H18_JOB_KEY',rid),event=event,**kw))
def wait_gpu():
 g=os.environ['GPU_UUID']
 while True:
  apps=subprocess.check_output(['nvidia-smi','-i',g,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  free=int(subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True))
  foreign=[p.strip() for p in apps.splitlines() if p.strip() and p.strip()!=str(os.getpid())]
  if not foreign and free>=21000:return
  time.sleep(5)
def run(cmd,path):
 wait_gpu()
 with Path(path).open('x') as f:return subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')).returncode
