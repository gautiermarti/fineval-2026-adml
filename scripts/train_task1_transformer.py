import json, random, re, sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from scripts.task1_numeric_baseline import parse_answer, score

MODEL = os.environ.get("TASK1_MODEL", "nlpaueb/bert-base-greek-uncased-v1")
OUTDIR = os.environ.get("TASK1_OUTDIR", "artifacts/task1_transformer")
LABELS = ["O"] + [f"{p}-{t}" for t in ("ΧΡΗΜΑΤΑ","ΠΟΣΟΣΤΑ","ΧΡΟΝΙΚΑ","ΠΟΣΟΤΗΤΕΣ","ΑΛΛΑ") for p in ("B","I")]
L2I = {x:i for i,x in enumerate(LABELS)}


def align_annotations(text, spans):
    """Align an unordered answer multiset to distinct occurrences in text."""
    annotations=[]; used=[]
    for entity, entity_type in spans:
        pattern=re.escape(entity)
        if entity and entity[0].isdigit(): pattern=r'(?<!\d)'+pattern
        if entity and entity[-1].isdigit(): pattern=pattern+r'(?!\d)'
        candidates=[match.span() for match in re.finditer(pattern,text)]
        chosen=next((span for span in candidates if not any(
            span[0] < right and span[1] > left for left,right in used
        )),None)
        if chosen is None:
            continue
        annotations.append((*chosen,entity_type)); used.append(chosen)
    return sorted(annotations)

def make_examples(frame, tok):
    out=[]
    for _, row in frame.iterrows():
        text=str(row.text); spans=parse_answer(row.answer)
        anns=align_annotations(text,spans)
        enc=tok(text,return_offsets_mapping=True,truncation=True,max_length=512)
        ys=[]
        for a,b in enc.offset_mapping:
            lab="O"
            if b>a:
                for s,e,t in anns:
                    if a>=s and b<=e: lab=("B-" if a==s else "I-")+t; break
            ys.append(L2I[lab] if lab in L2I else 0)
        out.append((enc,ys,text,spans))
    return out

class DS(Dataset):
    def __init__(self,x): self.x=x
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        e,y,_,_=self.x[i]; return {k:torch.tensor(v) for k,v in e.items() if k!='offset_mapping'}|{'labels':torch.tensor(y)}

def collate(batch):
    n=max(x['input_ids'].numel() for x in batch); out={}
    for k in batch[0]:
        vals=[]
        for x in batch:
            z=x[k]; vals.append(torch.nn.functional.pad(z,(0,n-z.numel()),value=-100 if k=='labels' else 0))
        out[k]=torch.stack(vals)
    return out

def decode(logits, examples, tok):
    result=[]
    for logit,(enc,_,text,_) in zip(logits,examples):
        pred=logit.argmax(-1).tolist(); ents=[]; cur=None
        for (a,b),lab in zip(enc.offset_mapping,pred):
            if b<=a or lab==0:
                continue
            name=LABELS[lab]; pref,typ=name.split('-',1)
            if pref=='B' or cur is None or cur[2]!=typ:
                if cur: ents.append(cur)
                cur=[a,b,typ]
            else: cur[1]=b
        if cur: ents.append(cur)
        result.append([(text[a:b],t) for a,b,t in ents])
    return result

def main():
    seed=int(os.environ.get('TASK1_SEED','7')); random.seed(seed); torch.manual_seed(seed)
    tr=pd.read_parquet('data/raw/subtask1/train-00000-of-00001.parquet'); va=pd.read_parquet('data/raw/subtask1/validation-00000-of-00001.parquet')
    tok=AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    a=make_examples(tr,tok); b=make_examples(va,tok)
    model=AutoModelForTokenClassification.from_pretrained(MODEL,num_labels=len(LABELS),id2label=dict(enumerate(LABELS)),label2id=L2I,ignore_mismatched_sizes=True)
    dev='cuda' if torch.cuda.is_available() else 'cpu'; model.to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
    dl=DataLoader(DS(a),batch_size=8,shuffle=True,collate_fn=collate)
    best=0
    for ep in range(8):
        model.train()
        for batch in dl:
            batch={k:v.to(dev) for k,v in batch.items()}; loss=model(**batch).loss; loss.backward(); opt.step(); opt.zero_grad()
        model.eval(); preds=[]
        with torch.no_grad():
            for i in range(0,len(b),8):
                xs=b[i:i+8]; batch=collate([DS(b)[j] for j in range(i,min(i+8,len(b)))])
                preds += decode(model(input_ids=batch['input_ids'].to(dev),attention_mask=batch['attention_mask'].to(dev)).logits.cpu(),xs,tok)
        f=sum(score(parse_answer(r.answer),p)[2] for r,p in zip(va.itertuples(),preds))/len(va)
        print(ep+1,f)
        if f>best:
            best=f; Path(OUTDIR).mkdir(parents=True,exist_ok=True)
            model.save_pretrained(OUTDIR); tok.save_pretrained(OUTDIR)
            json.dump(
                {'f1':f,'model':MODEL,'best_epoch':ep+1,'seed':seed},
                open(Path(OUTDIR)/'metrics.json','w'),
            )

if __name__=='__main__': main()
