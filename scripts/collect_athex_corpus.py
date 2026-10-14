"""Resumable collector for public Greek ATHEX issuer financial reports."""
import re, requests, urllib.parse, time, os, json, hashlib
from pathlib import Path

ROOT=Path('data/raw/finance_corpus'); OUT=ROOT/'text'/os.environ.get('ATHEX_OUT','athex_issuers'); OUT.mkdir(parents=True,exist_ok=True)
S=requests.Session(); S.headers['User-Agent']='fineval-research-corpus/1.0'
ERROR_MARKERS = (
    'Target URL returned error', 'Title: Η σελίδα δε βρέθηκε',
    'Page Not Found', '404: Not Found',
)
def get(url,timeout=90):
    return S.get(url,timeout=timeout).text


def is_greek_financial_report(text):
    """Reject large proxy/error pages and non-Greek non-financial payloads."""
    if len(text) <= 10_000 or any(marker in text for marker in ERROR_MARKERS):
        return False
    greek = sum(
        '\u0370' <= char <= '\u03ff' or '\u1f00' <= char <= '\u1fff'
        for char in text
    )
    financial = len(re.findall(
        r'οικονομικ|χρηματοοικονομ|ισολογισ|κατάσταση|έκθεση|έσοδ|κέρδ',
        text, re.IGNORECASE,
    ))
    return greek >= 1_000 and financial >= 4
def main():
    # ATHEX loads the complete directory from this public JSON data file;
    # scraping the visible HTML only exposes the first page.
    directory=S.get(os.environ.get('ATHEX_DIRECTORY','https://www.athexgroup.gr/sites/default/files/json_data_files/issuers_el.json'),timeout=90).json()
    ids=[str(x['Code']) for x in directory.get('data',[]) if x.get('Code')]
    log=ROOT/'athex_collection_progress.jsonl'
    for n,iid in enumerate(ids):
        try: page=get(f'https://r.jina.ai/https://www.athexgroup.gr/el/market-data/issuers/{iid}/financial-data')
        except Exception as e: print('PAGE',iid,e,flush=True); continue
        urls=[]
        for u in re.findall(r'https?[^ )]+?\.pdf(?:\?[^ )]*)?',page):
            u=urllib.parse.unquote(u)
            if u not in urls: urls.append(u)
        for j,u in enumerate(urls):
            if os.environ.get('ATHEX_CURRENT_ONLY') == '1' and 'athens.euronext.com/' not in u:
                continue
            # Page position is not stable: new filings reorder the issuer page.
            # URL-addressed names prevent a later crawl from overwriting history.
            url_key = hashlib.sha256(u.encode()).hexdigest()[:16]
            out=OUT/f'{iid}_{url_key}.txt'
            if out.exists() and out.stat().st_size>10000: continue
            try:
                z=get('https://r.jina.ai/http://'+u.split('://',1)[-1],timeout=120)
                if is_greek_financial_report(z):
                    out.write_text(z,encoding='utf-8')
                    with log.open('a') as h: h.write(json.dumps({
                        'issuer':iid, 'index':j, 'url':u, 'bytes':len(z),
                        'output':str(out), 'sha256':hashlib.sha256(z.encode()).hexdigest(),
                    },ensure_ascii=False)+'\n')
            except Exception as e: print('DOC',iid,j,e,flush=True)
        if n%10==0: print('ISSUER',n+1,'/',len(ids),'id',iid,'docs',len(urls),flush=True)
    print('DONE',len(ids),flush=True)
if __name__=='__main__': main()
