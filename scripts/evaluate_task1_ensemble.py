import sys, json, os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]))
import pandas as pd, torch
from scripts.train_task1_transformer import MODEL, LABELS, make_examples, decode, score, parse_answer, collate
from scripts.task1_postprocess import add_explicit_identifiers
from transformers import AutoTokenizer, AutoModelForTokenClassification

def main():
    frame=pd.read_parquet('data/raw/subtask1/validation-00000-of-00001.parquet')
    paths=os.environ.get('TASK1_ENSEMBLE_PATHS','artifacts/task1_legal_base,artifacts/task1_legal_bert_seed7').split(',')
    dev='cuda' if torch.cuda.is_available() else 'cpu'; outputs=[]; examples=None
    for path in paths:
        tok=AutoTokenizer.from_pretrained(path); examples=make_examples(frame,tok)
        model=AutoModelForTokenClassification.from_pretrained(path).to(dev).eval(); logits=[]
        with torch.no_grad():
            for ex in examples:
                ids=torch.tensor(ex[0]['input_ids'])[None].to(dev); mask=torch.tensor(ex[0]['attention_mask'])[None].to(dev)
                logits.append(model(input_ids=ids,attention_mask=mask).logits[0].cpu())
        outputs.append(logits)
    candidates=[]
    if os.environ.get('TASK1_GRID') and len(outputs)==3:
      for a in [0,.1,.2,.3,.4]:
       for b in [0,.1,.2,.3,.4]:
        c=1-a-b
        if c>=0: candidates.append([a,b,c])
    else: candidates=[[float(x) for x in os.environ.get('TASK1_ENSEMBLE_WEIGHTS',','.join(['1']*len(outputs))).split(',')]]
    best=(-1,None,None)
    for weights in candidates:
      z=sum(weights) or 1.; weights=[x/z for x in weights]
      avg=[sum((weights[k]*outputs[k][i] for k in range(len(outputs))), torch.zeros_like(outputs[0][i])) for i in range(len(examples))]
      pred=decode(avg,examples,tok)
      if os.environ.get('TASK1_POSTPROCESS_IDS'):
       pred=[add_explicit_identifiers(r.text,p) for r,p in zip(frame.itertuples(),pred)]
      f=sum(score(parse_answer(r.answer),p)[2] for r,p in zip(frame.itertuples(),pred))/len(frame)
      if f>best[0]: best=(f,weights,pred)
    f,weights,pred=best; print('ensemble_f1',f,'weights',weights)
    Path('artifacts/task1_ensemble').mkdir(exist_ok=True)
    with open('artifacts/task1_ensemble/validation_predictions.jsonl','w') as h:
      for i,(p,r) in enumerate(zip(pred,frame.itertuples())):
       h.write(json.dumps({'row_index':i,'entities':p,'f1':score(parse_answer(r.answer),p)[2]},ensure_ascii=False)+'\n')
if __name__=='__main__': main()
