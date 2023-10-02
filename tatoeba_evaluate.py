import faiss
import torch
import numpy as np
import os
import logging
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm
# import faiss
from typing import List, AnyStr
from collections import Counter
import tatoeba_dataset
from evaluate import load_model, MODEL_PATH, run_inference
from evaluate_tools import simple_search

logger = logging.getLogger(__name__)
lang3_dict = {'ara': 'ar', 'heb': 'he', 'vie': 'vi', 'ind': 'id',
              'jav': 'jv', 'tgl': 'tl', 'eus': 'eu', 'mal': 'ml', 'tam': 'ta',
              'tel': 'te', 'afr': 'af', 'nld': 'nl', 'eng': 'en', 'deu': 'de',
              'ell': 'el', 'ben': 'bn', 'hin': 'hi', 'mar': 'mr', 'urd': 'ur',
              'tam': 'ta', 'fra': 'fr', 'ita': 'it', 'por': 'pt', 'spa': 'es',
              'bul': 'bg', 'rus': 'ru', 'jpn': 'jp', 'kat': 'ka', 'kor': 'ko',
              'tha': 'th', 'swh': 'sw', 'cmn': 'zh', 'kaz': 'kk', 'tur': 'tr',
              'est': 'et', 'fin': 'fi', 'hun': 'hu', 'pes': 'fa', 'aze': 'az',
              'lit': 'lt', 'pol': 'pl', 'ukr': 'uk', 'ron': 'ro'}
def logger_init():
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')

    handler = logging.FileHandler("./tatoeba_log.txt")
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(console)
    logger.addHandler(handler)

logger_init()

acc_dir = './output/accuracy/'

def local_layer_choose(model_name='ERNIE-M', bi_mean=True, lang_layer_acc=None):
    acc = []
    lang_names = []
    acc_dir = f'./output/accuracy_{model_name}/'
    for f in os.listdir(acc_dir):
        lang = f.split('.')[0].split('_')[-1]
        acc_lang = np.load(acc_dir + f)
        if bi_mean:
            acc_lang = np.mean((acc_lang[:13, 0], acc_lang[13:, 0]), axis=0)
            # print(acc_lang.shape)
            # return
        else:
            # en - x
            acc_lang = acc_lang[13:, 0]
        acc.append(acc_lang)
        lang_names.append(lang)
    acc = np.stack(acc)

    assert acc.shape == (112, 13) or acc.shape == (112, 24), "acc should be [LANGS, LAYERS]"
    # print(acc.shape)
    layer_wise_acc = acc.mean(axis=0)
    # print(layer_wise_acc.shape)
    best_layer = np.argmax(layer_wise_acc)
    print(f"layer{best_layer} have best avg acc of all language: {layer_wise_acc[best_layer]}")
    #
    # optimal_res = np.max(acc, axis=1)
    optimal_layers = np.argmax(acc, axis=1)
    # print(f"optimal avg acc {optimal_res.mean()}")
    # # print(optimal_res, acc[:, i])
    # print("optimal layer count: ", Counter(optimal_layers))

    print('for inconsistency:')
    for i, x in enumerate(optimal_layers):
        if x != best_layer:
            print('%s, %.4f, %.4f, %.4f' % (lang_names[i], acc[i][x], acc[i][best_layer], acc[i][x] - acc[i][best_layer]))

def layer_choose(model_name='ERNIE-M', bi_mean=True, lang_layer_acc=None):
    # output/accuracy (13, 2) 13layers *  [x-en, en-x]
    if lang_layer_acc is not None:
        acc = lang_layer_acc
    else:
        acc_file = f'./output/tatoeba/{model_name}_all_acc.npy'
        if bi_mean:
            acc_all = np.mean(np.load(acc_file), axis=2)
        else:
            acc_all = np.load(acc_file)[:, :, 1]
        acc = acc_all

    assert acc.shape == (112, 13) or acc.shape == (112, 24), "acc should be [LANGS, LAYERS]"
    # print(acc.shape)
    layer_wise_acc = acc.mean(axis=0)
    # print(layer_wise_acc.shape)
    i = np.argmax(layer_wise_acc)
    print(f"layer{i} have best avg acc of all language: {layer_wise_acc[i]}")

    optimal_res = np.max(acc, axis=1)
    optimal_layer = np.argmax(acc, axis=1)
    print(f"optimal avg acc {optimal_res.mean()}")
    # print(optimal_res, acc[:, i])

    print("optimal layer count: ", Counter(optimal_layer))

def encode_1(model_name, lang, layer=13):
    pool_type = 'mean'
    batch_size = 1024
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config, model, tokenizer = load_model(model_path=MODEL_PATH[model_name], device=device, output_hidden_states=True)
    layer = config.num_hidden_layers + 1

    print(f'embedding {lang} in {model_name}')
    txt_lang, txt_eng = tatoeba_dataset.load_sents(lang)
    embed_lang = run_inference(txt_lang, model_name, config, model, tokenizer, pool_type, batch_size)
    embed_eng = run_inference(txt_eng, model_name, config, model, tokenizer, pool_type, batch_size)

    for i in range(layer):
        tatoeba_dataset.save_embeds(embed_lang[i], embed_eng[i], model_name, lang, i)

    return

def extract_all_embedding(model_name, langs = None, pool_type='mean', batch_size=32, to_en=True, save=True):
    if langs is None:
       langs = tatoeba_dataset.load_langs()
    # langs = ['deu', 'fra', 'jpn', 'cmn']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    config, model, tokenizer = load_model(model_path=MODEL_PATH[model_name], device=device, output_hidden_states=True)
    layer = config.num_hidden_layers + 1

    for lang in langs:
        print(f'embedding {lang} in {model_name}')
        txt_lang, txt_eng = tatoeba_dataset.load_sents(lang)
        embed_lang = run_inference(txt_lang, model_name, config, model, tokenizer, pool_type, batch_size, to_en=to_en)
        embed_eng = run_inference(txt_eng, model_name, config, model, tokenizer, pool_type, batch_size, to_en=to_en)

        if save:
            for i in range(layer):
                tatoeba_dataset.save_embeds(embed_lang[i], embed_eng[i], model_name, lang, i)

    return


def evaluate_acc(model_name, langs=None, layer=13, cosine=False):
    if langs is None:
        langs = tatoeba_dataset.load_langs()
    gpu = False
    acc = []

    for lang in langs:
        acc_layers = []
        print(f'calculate acc in {lang} in {model_name} layer{range(layer)}')
        for i in range(13):
            embed_lang, embed_eng = tatoeba_dataset.load_embeds(model_name, lang, layer=i)
            index = simple_search(embed_eng, embed_lang, cosine=cosine, gpu=gpu)
            hit = [1 if index[i] == i else 0 for i in range(index.shape[0])]
            acc_lang = np.mean(hit)

            index = simple_search(embed_lang, embed_eng, cosine=cosine, gpu=gpu)
            hit = [1 if index[i] == i else 0 for i in range(index.shape[0])]
            acc_eng = np.mean(hit)

            acc_layers.append([acc_lang, acc_eng])

        acc.append(acc_layers)


    res = np.array(acc)
    logger.info(f'have all acc: {res.shape}')
    np.save(f'./output/tatoeba/{model_name}_all_acc.npy', res)
    return res


def evaluate_acc_1(model_name, lang, layer=13, cosine=True):
    gpu = False
    acc_layers = []
    for i in range(7, 8):
        embed_lang, embed_eng = tatoeba_dataset.load_embeds(model_name, lang, layer=i)
        index = simple_search(embed_eng, embed_lang, cosine=cosine, gpu=gpu)
        print(index)
        hit = [1 if index[i] == i else 0 for i in range(index.shape[0])]
        acc_lang = np.mean(hit)

        index = simple_search(embed_lang, embed_eng, cosine=cosine, gpu=gpu)
        hit = [1 if index[i] == i else 0 for i in range(index.shape[0])]
        acc_eng = np.mean(hit)

        acc_layers.append([acc_lang, acc_eng])

    print(acc_layers)





if __name__ == '__main__':
    # encode_1('XLM-R', 'fra')

    # evaluate_acc_1('XLM-R', 'fra')
    # local_layer_choose('XLM-A', bi_mean=True, lang_layer_acc=None)

    #
    # model_name = 'XLM-R'
    # print(np.load(f'./output/tatoeba/{model_name}_all_acc.npy'))
    langs = ['deu', 'fra', 'jpn', 'cmn']

    for model_name in ['XLM-R']:
        extract_all_embedding(model_name, pool_type='mean', langs=langs, batch_size=32, to_en=True, save=True)
        lang_layer_acc = evaluate_acc(model_name, langs=langs, cosine=False)
        print(np.max(lang_layer_acc, axis=(1, 2)))

    # for model_name in ['XLM-R']:
    #     extract_all_embedding(model_name, pool_type='mean', langs=langs, batch_size=32, to_en=True, save=True)
    #     lang_layer_acc = evaluate_acc(model_name, langs=langs, cosine=False)
    #     print(lang_layer_acc)
        # layer_choose(model_name, bi_mean=True)

    # for model_name in ['INFOXLM']:
    #     lang_layer_acc = evaluate_acc(model_name, False)
    #     layer_choose(model_name, bi_mean=True, lang_layer_acc=lang_layer_acc)