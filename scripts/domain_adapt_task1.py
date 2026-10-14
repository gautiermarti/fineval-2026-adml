import os, random, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling

BASE=os.environ.get('TASK1_BASE','spyrosbriakos/greek_legal_bert_v2'); OUT=os.environ.get('TASK1_ADAPT_OUT','artifacts/task1_domain_adapted_clean'); CORPUS=Path(os.environ.get('TASK1_CORPUS','data/processed/finance_corpus_clean'))
class TextDS(Dataset):
 def __init__(self, tok, files):
  self.rows=[]
  for f in files:
   s=Path(f).read_text(errors='ignore')
   words=s.split()
   for i in range(0,len(words),180):
    t=' '.join(words[i:i+180])
    if len(t)>100:self.rows.append(tok(t,truncation=True,max_length=256))
 def __len__(self): return len(self.rows)
 def __getitem__(self,i): return {k:torch.tensor(v) for k,v in self.rows[i].items()}
def main():
 random.seed(7); torch.manual_seed(7); dev='cuda' if torch.cuda.is_available() else 'cpu'
 tok=AutoTokenizer.from_pretrained(BASE); files=list(CORPUS.rglob('*.txt')); ds=TextDS(tok,files); print('examples',len(ds),'device',dev,flush=True)
 coll=DataCollatorForLanguageModeling(tok,mlm_probability=.15); dl=DataLoader(ds,batch_size=16,shuffle=True,collate_fn=coll)
 model=AutoModelForMaskedLM.from_pretrained(BASE).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=5e-5)
 model.train()
 for ep in range(2):
  total=0
  for n,b in enumerate(dl):
   b={k:v.to(dev) for k,v in b.items()}; loss=model(**b).loss; loss.backward(); opt.step(); opt.zero_grad(); total+=loss.item()
   if n%50==0: print(ep,n,loss.item(),flush=True)
  print('epoch',ep,'mean',total/len(dl),flush=True)
 Path(OUT).mkdir(parents=True,exist_ok=True); model.save_pretrained(OUT); tok.save_pretrained(OUT)
if __name__=='__main__': main()
