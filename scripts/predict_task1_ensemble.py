#!/usr/bin/env python3
"""Generate Task1 entity,type answers with the frozen robust ensemble."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from scripts.train_task1_transformer import make_examples, decode
from scripts.task1_postprocess import add_explicit_identifiers, add_zone_designators

DEFAULT_MODELS = [
    'artifacts/task1_legal_base',
    'artifacts/task1_legal_bert_seed7',
    'artifacts/task1_domain_finetuned',
]
FINAL_REFIT_MODELS = [
    'artifacts/task1_full_refit/legal_base',
    'artifacts/task1_full_refit/legal_ner_init',
    'artifacts/task1_full_refit/finance_adapted',
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--models',nargs='+'); ap.add_argument('--final-refit',action='store_true'); ap.add_argument('--zone-designators',action='store_true'); args=ap.parse_args()
    model_paths=args.models or (FINAL_REFIT_MODELS if args.final_refit else DEFAULT_MODELS)
    frame=pd.read_parquet(args.input); dev='cuda' if torch.cuda.is_available() else 'cpu'; outputs=[]; examples=None; tok=None
    for path in model_paths:
        tok=AutoTokenizer.from_pretrained(path); examples=make_examples(frame.assign(answer=''),tok); model=AutoModelForTokenClassification.from_pretrained(path).to(dev).eval(); rows=[]
        with torch.no_grad():
            for ex in examples:
                ids=torch.tensor(ex[0]['input_ids'])[None].to(dev); mask=torch.tensor(ex[0]['attention_mask'])[None].to(dev)
                rows.append(model(input_ids=ids,attention_mask=mask).logits[0].cpu())
        outputs.append(rows)
    avg=[sum((outputs[k][i] for k in range(len(outputs))),torch.zeros_like(outputs[0][i]))/len(outputs) for i in range(len(frame))]
    pred=decode(avg,examples,tok); pred=[add_explicit_identifiers(r.text,p) for r,p in zip(frame.itertuples(),pred)]
    if args.zone_designators:
        pred=[add_zone_designators(r.text,p) for r,p in zip(frame.itertuples(),pred)]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w') as h:
        for i,p in enumerate(pred):
            answer='\n'.join(f'{entity}, {label}' for entity,label in p)
            h.write(json.dumps({'row_index':i,'answer':answer},ensure_ascii=False)+'\n')
    print(f'wrote {len(pred)} predictions to {args.output}')
if __name__=='__main__': main()
