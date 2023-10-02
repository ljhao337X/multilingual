import faiss
import numpy as np
from bucc_dataset import load_bucc_embeddings, LAYER_LIST, Gold
import pickle
import logging
logger = logging.getLogger(__name__)

def EmbedLoad(fname, dim=1024, verbose=True, fp16=False):
    x = np.fromfile(fname, dtype=(np.float16 if fp16 else np.float32), count=-1)
    x.resize(x.shape[0] // dim, dim)
    if verbose:
        print(" - Embeddings: {:s}, {:d}x{:d}".format(fname, x.shape[0], dim))
    return x

def TextLoadUnify(fname, unify=True, verbose=True):
    if verbose:
        print(' - loading texts {:s}: '.format(fname), end='')
    fin = open(fname, encoding='utf-8', errors='surrogateescape')
    inds = []
    sents = []
    sent2ind = {}
    duplicate = []
    n = 0
    nu = 0
    for line in fin:
        line = line.split("\t")[1]
        new_ind = len(sent2ind)
        inds.append(sent2ind.setdefault(line, new_ind))
        if unify:
            if inds[-1] == new_ind:
                sents.append(line[:-1])
                nu += 1
            else:
                # has duplicate
                duplicate.append([sent2ind[line], n+1])
        else:
            sents.append(line[:-1])
            nu += 1
        n += 1
    if verbose:
        print('{:d} lines, {:d} unique'.format(n, nu))
    del sent2ind
    return inds, sents, duplicate


def simple_search(query, doc, cosine=False, gpu=True):
    if cosine:
        faiss.normalize_L2(doc)
        faiss.normalize_L2(query)
     # print(f'query sent_num:{query.shape[0]}, doc sent_num:{doc.shape[0]}')

    dimension = doc.shape[1]
    # build a flat (CPU) index
    # cosine使用inner product
    if cosine:
        index_flat = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(doc)
        faiss.normalize_L2(query)
    else:
        index_flat = faiss.IndexFlatL2(dimension)
    # make it into a gpu index
    if gpu:
        resource = faiss.StandardGpuResources()  # use a single GPU, 这个命令需要安装Faiss GPU 版本
        index_flat = faiss.index_cpu_to_gpu(resource, 0, index_flat)
    index_flat.add(doc)  # add vectors to the index
    distance, index = index_flat.search(query, 1)
    # print(distance[:5], index[:5])

    # generate result
    index = index.astype(int)
    return index


def generate_margin_k_similarity(query, doc,  k=4, margin=True, cosine=True, gpu=True, reverse_index=False):
    '''
    :param query: default lang
    :param doc: defualt en
    :param k: use with marigin
    :param cosine:
    :param gpu:
    :param reverse_index: if True q->en, doc->lang
    :return:
    '''
    # assert k==1 or (margin and k>1), print('if use margin calulation, k>1 is needed')
    # cosine 需要l2 norm
    print(f'query sent_num:{query.shape[0]}, doc sent_num:{doc.shape[0]}, margin:{k}')

    dimension = doc.shape[1]
    
    if cosine:
        faiss.normalize_L2(doc)
        faiss.normalize_L2(query)
        index_flat = faiss.IndexFlatIP(dimension)
    else:
        index_flat = faiss.IndexFlatL2(dimension)
    
    # make it into a gpu index
    if gpu:
        resource = faiss.StandardGpuResources()  # use a single GPU, 这个命令需要安装Faiss GPU 版本
        index_flat = faiss.index_cpu_to_gpu(resource, 0, index_flat)
    index_flat.add(doc)  # add vectors to the index
    
    if not margin:
        k = 1
    distance, index = index_flat.search(query, k)

    # generate result
    index = index.astype(int)
    if margin:
        if cosine:
            # distance is inner product ~ cosine ~ similarity
             similarity = (distance[:, 0] - np.sum(distance, axis=1, keepdims=False) / k)
        else:
            # distance is l2
            # distance_k / distance_0 ~ how close 0 is ~ similarity
             similarity = (np.sum(distance, axis=1, keepdims=False) / k) / distance[:, 0]
    else:
        if cosine:
             similarity = distance[:, 0]
        else:
             similarity = - distance[:, 0]

    res = []
    if reverse_index:
        for i in range(similarity.shape[0]):
            res.append([index[i, 0]+1, i+1, float(similarity[i])])
    else:
        for i in range(similarity.shape[0]):
            res.append([i+1, index[i, 0]+1, float(similarity[i])])

    return res

def bucc_optimize(candidate_and_similarity, gold: Gold):
    '''
    :param candidate_and_similarity: [sent_num, 3] [[index_q, index_doc, sim]]
    :param gold: dict
    :return: (best) f1, threshold, recall(independent), precision(along with f1)
    '''
    # 相似度从大到小排序
    print(f'total candidates: {len(candidate_and_similarity)}')
    gold_count = gold.gold_count()
    items = sorted(candidate_and_similarity, key=lambda x: -x[2])
    # print(items[0:5])
    positive = true_positive = 0
    threshold = 0.0
    best_f1 = 0.0
    true_positive_count = 0.0
    best_precision = 0.0
    best_recall = 0.0
    for x in items:
        positive += 1

        if gold.is_gold(x[0], x[1]):
            true_positive += 1
            true_positive_count += 1
            # print('a true positive')
        if true_positive > 0:
            precision = true_positive / positive
            recall = true_positive / gold_count
            f1 = 2 * precision * recall / (precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                threshold = x[2]
                best_precision = precision

            if recall > best_recall:
                best_recall = recall
    return best_f1, threshold, best_recall, best_precision


def bucc_f1(candidate_and_similarity, gold):
    # 相似度从小到大排序
    ngold = len(gold)
    positive = true_positive = 0
    threshold = 0
    best_f1 = 0
    for x in candidate_and_similarity:
        if x[2] >= threshold:
            positive += 1
            if x[1] == gold[x[0]]:
                true_positive += 1

    precision = true_positive / positive
    recall = true_positive / ngold
    f1 = 2 * precision * recall / (precision + recall)
    return f1


if __name__ == '__main__':
    def unique_embeddings(emb, ind, verbose=False):
        aux = {j: i for i, j in enumerate(ind)}
        if verbose:
            print(' - unify embeddings: {:d} -> {:d}'.format(len(emb), len(aux)))
        return emb[[aux[i] for i in range(len(aux))]]

    gold = Gold('zh', 'training')
    # x = EmbedLoad('./dataset/bucc/bucc2018/zh-en/laser_en_sample_5.npy', fp16=False).astype(np.float32)
    # x = unique_embeddings(x, src_inds)

    # model_name = 'XLM-R'
    # data_type = 'sample'
    # for language in ['zh']:
    # # for language in ['zh', 'fr', 'de', 'ru']:
    #     for i in [1, 5, 7, 8, 9, 12]:
    #         print(f'\n evaluate on {model_name}, layer-{i} in {language}-en setting')
    #         embeddings_lang, embeddings_en = load_bucc_embeddings(language, model_name, i, data_type=data_type)
    #         lang_en_dict, en_lang_dict = get_bucc_gold(language=language, data_type=data_type)
    #
    #         candidate_and_score_en_to_lang = generate_margin_k_similarity(embeddings_lang, embeddings_en, cosine=False)
    #         # with open(f'./dataset/bucc/bucc2017/{language}-en/training-pair-similarity.pickle', 'wb') as f:
    #         #     pickle.dump(candidate_and_score, f)
    #
    #         # print(candidate_and_score)
    #         threshold = bucc_optimize(candidate_and_score_en_to_lang, lang_en_dict)

