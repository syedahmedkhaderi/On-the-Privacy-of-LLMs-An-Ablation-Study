"""Runs dea_guess_control.py UNCHANGED (sha checked). Adds the verified VM2 copies (vm2-dea-detections/) to the cell search by wrapping glob.glob, and renames the aggregate to *-v3.json."""
import os,sys,glob,hashlib
F="work/cyclea-protected/"; S=F+"analysis/work/dea_guess_control.py"
src=open(S,"rb").read(); assert hashlib.sha256(src).hexdigest()=="93cc173839ad69c649958dfdcdd3ea5ecfeaa46eba7533c332f84c32c14fa550"
_o=os.open; _g=glob.glob
def op(p,*a,**k):
    if p==F+"aggregates/dea-guess-control.json": p=F+"aggregates/dea-guess-control-v3.json"
    return _o(p,*a,**k)
def gl(pat,*a,**k):
    r=_g(pat,*a,**k)
    if pat.startswith(F+"runs/ca-deab2-"): r=r+_g(pat.replace(F+"runs/",F+"vm2-dea-detections/",1),*a,**k)
    return r
os.open=op; glob.glob=gl
exec(compile(src,S,"exec"),{"__name__":"__main__"})
