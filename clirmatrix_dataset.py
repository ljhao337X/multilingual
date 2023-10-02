import json
import gzip
import sys
import glob
from tqdm import tqdm
from clirmatrix_evaluate import idcg
import numpy as np

_gold_dir = './dataset/CLIRMatrix/MULTI-8/'
_doc_dir = './dataset/CLIRMatrix/doc/'
_splits = ['train', 'dev', 'test1', 'test2']
_languages = ['ar', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'zh']



def unzip_gold(src_lang, tgt_lang, split):
    assert src_lang in _languages and tgt_lang in _languages and split in _splits
    f = _gold_dir + f'{src_lang}/{src_lang}.{tgt_lang}.{split}.jl.gz'
    limit = 5
    count = {}
    res = {}
    for i, jsonl in tqdm(enumerate(gzip.open(f, "rt", encoding='utf-8'))):
        jsonl = json.loads(jsonl)

        src_id = int(jsonl['src_id'])
        src_query = jsonl['src_query']
        targets = jsonl['tgt_results'] # len(target) = 100
        x = idcg(targets)
        target_rel = {}
        for x in targets:
            assert x[1] is int
            target_rel[int(x[0])] = x[1]


        # targets_count = len(targets)
        # len(target) = 100
        res[src_id] = {"idcg": x, 'target_rel': target_rel}
        if i > 100:
            print(count)

    return res



        # if i%2==0:
        #     if "index" in jsonl:
        #         doc_id = jsonl["index"]["_id"] if "index" in jsonl else None
        # else:
        #     doc_text = jsonl["text"]
        #     if doc_id is not None:
        #         print("%s\t%s"%(doc_id, doc_text.replace("\t","")))


def load_embeddings(model_name, lang, layer):

    return np.Array([])

if __name__ == '__main__':
    unzip_gold('zh', 'de', 'test1')
    unzip_gold('zh', 'ar', 'train')