from math import log2

from typing import Dict

from clirmatrix_dataset import unzip_gold
from multiprocessing import Pool
import numpy as np

embedding = np.Array([])

def simple_search(query, doc, k=10):
    '''
    :param query:
    :param doc:
    :param k:
    :return: list of [id, similarity]
    '''

    return [[1, 0.01]]


def ndcg(ranked_ids, rel_docs: Dict[int, int], n=10):
    # \sum (2^p_i  / log2(i+1))
    ans = 0
    # assert len(ranked_ids) == 10

    for i in range(1, n + 1):
        tgt_id = ranked_ids[i]
        ans += 2 ** rel_docs[tgt_id] / log2(i + 1)
    return ans


def idcg(rel_docs, n=10):
    rel_docs.sort(key=lambda x: -x[1])
    ans = 0
    for i in range(1, n+1):
        ans += 2 ** rel_docs[i-1][1] / log2(i+1)
    return ans

class QueryLanguage():
    def __init__(self, src_lang, tgt_languages, split):
        self.target = {}
        for lang in tgt_languages:
            self.target[lang] = unzip_gold(src_lang, lang, split)

    def evaluate(self, lang):
        with Pool() as p:
            # with Pool(4) as p: # 指定4个进程
            data_infos = p.map(load_culane_ann, lines)


def read_src_id(src_id, tgt_lang, ql: QueryLanguage):
    src_vec = embedding[src_id] # [765]
    tgt_ids = list(ql.target[tgt_lang].keys())
    tgt_vec = embedding[tgt_ids] # [100, 765]

    cos_sim = src_vec.dot(tgt_vec) / (np.linalg.norm(src_vec) * np.linalg.norm(tgt_vec))

    ranked_ids = sorted(range(len(cos_sim)), key=lambda k: cos_sim[k], reverse=True)

    x = ncg(ranked_ids, ql.target[tgt_lang])

    return ranked_ids

if __name__ == '__main__':
    pass


