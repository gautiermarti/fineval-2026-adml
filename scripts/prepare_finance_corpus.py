from pathlib import Path
import re, hashlib, json, os
src=Path(os.environ.get('TASK1_CORPUS_SOURCE','data/raw/finance_corpus/text')); out=Path(os.environ.get('TASK1_CLEAN_OUT','data/processed/finance_corpus_clean')); out.mkdir(parents=True,exist_ok=True)
seen=set(); manifest=[]
for f in sorted(src.rglob('*.txt')):
 s=f.read_text(errors='ignore')
 if len(s)<1000 or 'σελίδα δε βρέθηκε' in s or 'Target URL returned error 404' in s: continue
 lines=[]
 for line in s.splitlines():
  line=line.strip()
  if not line or line.startswith(('Title:','URL Source:','Warning:','Markdown Content:')): continue
  if line.startswith(('![','[Παράκαμψη')): continue
  line=re.sub(r'\s+',' ',line)
  if len(line)>=20: lines.append(line)
 text='\n'.join(lines); paras=[]
 for p in text.splitlines():
  key=re.sub(r'\W+','',p.casefold())
  if len(key)<30 or key in seen: continue
  seen.add(key); paras.append(p)
 text='\n'.join(paras)
 if len(text)<1000: continue
 rel=f.relative_to(src); dest=out/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(text)
 manifest.append({'source':str(f),'clean':str(dest),'chars':len(text),'sha256':hashlib.sha256(text.encode()).hexdigest()})
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('documents',len(manifest),'chars',sum(x['chars'] for x in manifest))
