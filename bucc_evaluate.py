import torch
import numpy as np
import os
import logging
from tqdm import tqdm
from typing import List, AnyStr
from evaluate import load_model, MODEL_PATH, cls_pool_embedding, mean_pool_embedding, run_inference
from bucc_dataset import get_bucc_sentence, save_embedding_file, load_bucc_embeddings, LAYER_LIST, Gold, filter_dup_embeddings, get_duplicate
from draw_pic import draw_bucc_pic, reduce_to_2d_pca, reduce_to_2d_tsne
from evaluate_tools import generate_margin_k_similarity, bucc_optimize
import argparse
import matplotlib.pyplot as plt
# from bucc_laser import embed_sentences


BATCH_SIZE = 32
gpu = True
device = 'cuda:0' if torch.cuda.is_available() and gpu else 'cpu'
MODEL_LAYER = {'XLM-R-LARGE': 25, 'XLM-R': 13, 'INFOXLM':13}
from bucc_dataset import LAYER_LIST

logger = logging.getLogger(__name__)
def logger_register():
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler("./bucc_log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.addHandler(handler)
    logger.info('\n a new try')
logger_register()

# --------------------------------pic---------------------------------------------
def save_pic(languages, model_name, layer, compression_method, debug=False):
    embeddings = []
    counts = []
    langs = []

    for language in languages:
        # [sent, d]
        embeddings_lang, embeddings_en = load_bucc_embeddings(language, model_name, layer)

        if embeddings_lang is None:
            logger.info(f'you should run extract_embedding first(language:{language}, model:{model_name})')
            return
        else:
            logger.info(f'loaded {model_name} embeded {language}-en bucc data of layer {layer}')

        if debug:
            rng = np.random.default_rng()
            embeddings_lang = rng.choice(embeddings_lang, 4000, replace=False)
            embeddings_en = rng.choice(embeddings_en, 1000, replace=False)

        embeddings += [embeddings_lang, embeddings_en]
        counts += [len(embeddings_lang), len(embeddings_en)]
        langs += [language, 'en']

    logger.info(f'loaded all embeddings of model:{model_name}, layer:{layer}, {langs}, {counts}')

    embeddings = np.concatenate(embeddings)
    if compression_method == 'pca':
        embeddings = reduce_to_2d_pca(embeddings)
    if compression_method == 'tsne':
        embeddings = reduce_to_2d_tsne(embeddings)

    image = draw_bucc_pic(model_name, embeddings, langs, counts, compression_method, layer, debug=debug,
                          width=1500, height=1500)
    folder = './output/bucc/'
    if debug:
        folder += 'test/'
    # 生成文件名
    filename = f'bucc_{model_name}_{compression_method}_{layer}.jpg'
    if folder is not None and len(folder) > 0:
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, filename)

    # save to file
    print(f"[{model_name}]: save {layer}_embeddings to {filename}...")
    image.save(filename, quality=95, optimize=True, progressive=True)
    del image


def evaluate_on_gold(model_name='XLM-R', language='zh', layers=13, data_type='training', debug=True):
    '''
    :param model_name:
    :param language:
    :return: 计算所有gold的指标在各个层的关系
    '''
    import seaborn as sns
    gold = Gold(language, data_type)
    if debug:
        layers = range(10, 11)
    else:
        layers = LAYER_LIST[layers]
        # f, axes = plt.subplots(2, 7, figsize=(14, 28), sharex=True)

    for layer in layers:
        similarity = []
        embeddings_lang, embeddings_en = load_bucc_embeddings(language, model_name, layer)
        max_index_lang = len(embeddings_lang)
        max_index_en = len(embeddings_en)
        for k, v in gold.gold_lang.items():
            if k >= max_index_lang:
                print(f'index of {language} exceed: {k}')
                if v >= max_index_en:
                    print(f'index of en exceed: {k}')
            elif v >= max_index_en:
                print(f'index of en exceed: {k}')
            else:
                sim = np.sqrt(np.sum((embeddings_lang[k] - embeddings_en[v])**2))
                similarity.append(sim)
        logger.info(f'layer:{layer} gold similarity histogram:{np.histogram(similarity)}')
        if not debug:
            plt.cla()
            sns.set_style('darkgrid')
            # ax = axes[layer//7, layer%7]
            # ax.set_title(f"layer{layer}")
            sns.histplot(similarity)
            plt.savefig(f'./output/bucc/gold_hist_{language}_{layer}.png', dpi=300)

    # if not debug:
    #     plt.savefig(f'./output/bucc/gold_hist_{language}.png', dpi=300)
    #     plt.close()
    # print(similarity)
    # similarity = np.random.laplace(loc=15, scale=3, size=500)
    # print(np.histograms(similarity))
    plt.cla()
    return


#---------------------------generate embeddings--------------------------------------------------
def extract_embeddings_save(model_name='XLM-R', languages=['zh'], pool_type='cls', data_type='training', debug=True):
    config, model, tokenizer = load_model(model_path=MODEL_PATH[model_name], device=device, output_hidden_states=True)
    
    for language in languages:
        sentences_lang, sentences_en = get_bucc_sentence(language=language, data_type=data_type)
        logger.info(f'read bucc {data_type} file {language}-en, total lines:{len(sentences_lang), len(sentences_en)}')
        all_embeds_lang = run_inference(sentences_lang, model_name, config, model, tokenizer, language, pool_type=pool_type, debug=debug)
        all_embeds_en = run_inference(sentences_en, model_name, config, model, tokenizer, 'en', pool_type=pool_type, debug=debug)

        file_folder = save_embedding_file(model_name=model_name, embeds_lang=all_embeds_lang, embeds_en=all_embeds_en, data_type=data_type, language=language)
        del all_embeds_lang, all_embeds_en

        logger.info(f'saved file to {file_folder}')
    return


def combining_candidate_score(candidate_score1, candidate_score2, pooling='max'):
    '''
    candidate_score: list of (id, pred_pair_id, score)
    return combined_candidate_score
    '''
    candidate_dict = {}
    new_pair_count = 0
    same_pair = 0
    update_pair = 0
    # initialize
    # [i, candidate, score]
    for x in candidate_score1:
        candidate_dict[x[0]] = [x[1], x[2]]

    for x in candidate_score2:
        if candidate_dict.get(x[0]):
            if candidate_dict[x[0]][1] == x[1]:
            # same candidate
                candidate_dict[x[0]][1] = (candidate_dict[x[1]][1] + x[2]) / 2 if pooling == 'pooling' else max(candidate_dict[x[1]][1], x[2])
                same_pair += 1
            else:
                if candidate_dict[x[0]][1] < x[2]:
                    candidate_dict[x[0]] = [x[1], x[2]]
                    update_pair += 1
                else:
                    pass # do not update
        else:
            candidate_dict[x[0]] = [x[1], x[2]]
            new_pair_count += 1
    logger.info(f'bi-direction has samp_pair: {same_pair}, update_pair: {update_pair}, new pair{new_pair_count}, total:{len(candidate_dict)}')

    candidate_and_score = []
    for x in candidate_dict.keys():
        candidate_and_score.append([x, candidate_dict[x][0], candidate_dict[x][1]])
    return candidate_and_score


def restore_index(lang_sent_id, en_sent_id, candidate_score):
    '''
        restore index to en_index
    '''
    # sent_id 从0开始
    # candidate 从1开始
    for i in range(len(candidate_score)):
        id1, id2, score = candidate_score[i]
        candidate_score[i] = [lang_sent_id[id1-1]+1, en_sent_id[id2-1]+1, score]
    return

def generate_best_f1(model_name, language, layer, data_type, gold=None, cosine=True, margin=True, pooling='pooling'):
    if gold is None:
        gold = Gold(language, data_type)
    if pooling:
        logger.info(f'evaluate on {model_name}, layer-{layer} in {language}-en bi-{pooling} setting')
    else:
        logger.info(f'evaluate on {model_name}, layer-{layer} in {language}-en setting')

    embeddings_lang, embeddings_en = load_bucc_embeddings(language, model_name, layer, data_type=data_type)
    lang_sent_id, en_sent_id = filter_dup_embeddings(embeddings_lang, embeddings_en, gold)
    candidate_and_score = generate_margin_k_similarity(embeddings_lang, embeddings_en, cosine=cosine, margin=margin)
    if not pooling:
        pass
    else:
        candidate_and_score_en_to_lang = generate_margin_k_similarity(embeddings_en, embeddings_lang, cosine=cosine, reverse_index=True)
        candidate_and_score = combining_candidate_score(candidate_and_score, candidate_and_score_en_to_lang, pooling='max')

    restore_index(lang_sent_id, en_sent_id, candidate_and_score)
    best_f1, threshold, best_recall, best_precision = bucc_optimize(candidate_and_score, gold, ngold=gold.count)
    logger.info(f'optimized bucc f1: {best_f1}, threshold: {threshold}, best_f1: {best_recall}, best_precision: {best_precision}')
    return best_f1, best_recall



def generate_best_f1_layers(model_name, language, layers, data_type, cosine=True, margin=True, pooling='pooling'):
    '''
        calculate best f1 given model\language\layer
    '''
    gold = Gold(language, data_type)
    res = []
    for i in layers:
        best_f1, best_recall = generate_best_f1(model_name, language, layer=i, data_type=data_type, gold=gold, cosine=cosine, margin=margin, pooling=pooling)
        res.append([best_f1, best_recall])
    return res



def generate_gold_combined(model_name, language, layers, data_type, cosine=True, margin=True, bidirection=True, pooling='pooling', end=False):
    combined_candidate = []
    dup_lang, dup_en = get_duplicate(language=language, data_type=data_type)
    gold = Gold(language=language, data_type=data_type)

    if pooling:
        logger.info(f'evaluate on {model_name}, layer-{layers} in {language}-en bi-{pooling} setting')
    else:
        logger.info(f'evaluate on {model_name}, layer-{layers} in {language}-en setting')

    for i in range(len(layers)):
        embeddings_lang, embeddings_en = load_bucc_embeddings(language, model_name, layers[i], data_type=data_type)
        lang_sent_id, en_sent_id = filter_dup_embeddings(embeddings_lang, embeddings_en, dup_lang, dup_en)
        candidate_and_score = generate_margin_k_similarity(embeddings_lang, embeddings_en, cosine=cosine)

        if not pooling:
            pass
        else:
            candidate_and_score_en_to_lang = generate_margin_k_similarity(embeddings_en, embeddings_lang, cosine=cosine, reverse_index=True)
            candidate_and_score = combining_candidate_score(candidate_and_score, candidate_and_score_en_to_lang, pooling='max')

        if not end:
            logger.info(f'layer-{layers[i]} in {language}-en bi-{pooling} setting')
            restore_index(lang_sent_id, en_sent_id, candidate_and_score)
            best_f1, threshold, true_positive_count, best_precision = bucc_optimize(candidate_and_score, gold, ngold=gold.count)
            logger.info(f'optimized bucc f1: {best_f1}, threshold: {threshold}, total true positive: {true_positive_count}, best_precision: {best_precision}')

            logger.info(f'layer-{layers[:i+1]} in {language}-en bi-{pooling} setting')
            combined_candidate = combining_candidate_score(candidate_and_score, combined_candidate)
            best_f1, threshold, true_positive_count, best_precision = bucc_optimize(combined_candidate, gold,ngold=gold.count)
            logger.info(f'optimized bucc f1: {best_f1}, threshold: {threshold}, total true positive: {true_positive_count}, best_precision: {best_precision}')
        else:
            combined_candidate = combining_candidate_score(candidate_and_score, combined_candidate)

    if end:
        logger.info(f'layer-{layers} in {language}-en bi-{pooling} setting')
        restore_index(lang_sent_id, en_sent_id, candidate_and_score)
        best_f1, threshold, true_positive_count, best_precision = bucc_optimize(combined_candidate, gold,
                                                                                ngold=gold.count)
        logger.info(
            f'optimized bucc f1: {best_f1}, threshold: {threshold}, total true positive: {true_positive_count}, best_precision: {best_precision}')



def combine_embeddings_search(model_name, language, layers, data_type, cosine=True, margin=True, bidirection=True, pooling='pooling', alpha = 0.02):
    gold = Gold(language, data_type)
    for i in range(len(layers)):
        if i == 0:
            embeddings_lang, embeddings_en = load_bucc_embeddings(language, model_name, layers[i], data_type=data_type)
        else:
            t_embeddings_lang, t_embeddings_en = load_bucc_embeddings(language, model_name, layers[i], data_type=data_type)
            embeddings_lang = embeddings_lang + alpha * t_embeddings_lang
            embeddings_en = embeddings_en + alpha * t_embeddings_en

    lang_sent_id, en_sent_id = filter_dup_embeddings(embeddings_lang, embeddings_en, gold)
    candidate_and_score = generate_margin_k_similarity(embeddings_lang, embeddings_en, cosine=cosine)
    if not pooling:
        pass
    else:
        candidate_and_score_en_to_lang = generate_margin_k_similarity(embeddings_en, embeddings_lang, cosine=cosine,
                                                                      reverse_index=True)
        candidate_and_score = combining_candidate_score(candidate_and_score, candidate_and_score_en_to_lang,
                                                        pooling='max')

    logger.info(f'embed added layer-{layers} in {language}-en bi-{pooling} setting')
    restore_index(lang_sent_id, en_sent_id, candidate_and_score)
    best_f1, threshold, true_positive_count, best_precision = bucc_optimize(candidate_and_score, gold,
                                                                            ngold=gold.count)
    logger.info(
        f'optimized bucc f1: {best_f1}, threshold: {threshold}, total true positive: {true_positive_count}, best_precision: {best_precision}')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--flag",
                        action="store_true",
                        help="Run or not.")
    parser.add_argument("--type",
                        default='training',
                        help='run on training data or sample data')

    args = parser.parse_args()
    debug = not args.flag
    data_type = args.type

    if debug:
        BATCH_SIZE = 32
    else:
        BATCH_SIZE = 256


    model_name = 'XLM-A'
    extract_embeddings_save(model_name=model_name, languages=['de'], data_type=data_type, pool_type='mean', debug=False)
    res = []
    for language in ['zh', 'fr', 'ru', 'de']:
        layer_res = generate_best_f1_layers(model_name, language, layers=[1, 5, 7, 8, 9, 12], data_type=data_type, pooling='max')
        res.append(layer_res)
    logger.info(res)

    # extract_embeddings(model_name=model_name, language='zh', debug=debug)
    # evaluate_on_gold(model_name, language='zh', layers=MODEL_LAYER[model_name], debug=debug)
    # for layer in layer_list[MODEL_LAYER[model_name]]:


    # for layer in LAYER_LIST[MODEL_LAYER[model_name]]:
    #     save_pic(['zh', 'fr', 'de', 'ru'], model_name=model_name, layer=layer, compression_method='pca', debug=True)
    # extract_embeddings(model_name)






# 这个代码是要改哪些地方吗，我看了发现有1.RFClassifier应该是用别的单模态模型组合起来 2.forward里面有个execute_drop暂时好像还没看到哪里用了，也没看明白到底是想实现dropout还是直接drop掉整个模态信息？ 3cca_core没看到哪里用了
