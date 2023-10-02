import shutil

import numpy as np
import os
import random

# 这里的中文是繁体
BUCC_LAN = ['zh', 'de', 'fr', 'ru']
LAYER_LIST = {13: [1, 5, 7, 8, 9, 12], 25: [1, 5, 10, 15, 20, 24]}


def _duplicate_id2first(file):
    sent2id = {} # first id of a sentence
    i = 1
    duplicate_i2first = {}
    while True:
        line = file.readline()
        if not line:
            break

        sent = line.split('\t')[1]
        if sent2id.get(sent):
            duplicate_i2first[i] = sent2id[sent]
        else:
            sent2id[sent] = i
        i += 1

    return duplicate_i2first


def _get_duplicate(language='zh', data_type='training'):
    file_handler_lang = open(
        f"./dataset/bucc/bucc2018/{language}-en/{language}-en.{data_type}.{language}", "r",
        encoding="utf-8")
    file_handler_en = open(
        f"./dataset/bucc/bucc2018/{language}-en/{language}-en.{data_type}.en", "r",
        encoding="utf-8")

    duplicate_lang = _duplicate_id2first(file_handler_lang)
    duplicate_en = _duplicate_id2first(file_handler_en)
    print(f'duplicate {language} sents: {len(duplicate_lang)}, en sents" {len(duplicate_en)}')
    return duplicate_lang, duplicate_en

# bucc中句子数量远多于gold
class Gold:
    def __init__(self, language, data_type):
        self.gold_lang = {}
        assert data_type == 'training' or data_type == 'sample' or data_type == 'test', 'data_type input wrong'
        file_handler = open(
            f"./dataset/bucc/bucc2018/{language}-en/{language}-en.{data_type}.gold", "r",
            encoding="utf-8")
        for line in file_handler:
            id1, id2 = line.split('\t')
            id1 = int(id1.split('-')[1])
            id2 = int(id2.split('-')[1])
            self.gold_lang[id1] = id2
        
        self.dup_lang, self.dup_en = _get_duplicate(language, data_type)
        
    def is_gold(self, id1, id2):
        if self.gold_lang.get(id1):
            return self.gold_lang.get(id1) == id2
        else:
            return False
    
    def gold_count(self):
        return len(self.gold_lang)
    
    
    

def get_bucc_sentence(language='zh', max_seq=None, data_type='training', debug=True):
    assert data_type == 'training' or data_type=='sample', print('data_type error')
    file_handler1 = open(f"./dataset/bucc/bucc2018/{language}-en/{language}-en.{data_type}.{language}", "r", encoding="utf-8")
    file_handler2 = open(f"./dataset/bucc/bucc2018/{language}-en/{language}-en.{data_type}.en", "r", encoding="utf-8")

    lang1 = []
    lang2 = []
    i = 1
    while True:
        if max_seq is not None:
            if len(lang1) >= max_seq:
                break
        line1 = file_handler1.readline()
        if not line1:
            break
        order1, line1 = line1.split('\t')
        order1 = int(order1.split('-')[1])
        if i == order1:
            i += 1
        else:
            print(order1, line1)
        # print(order1, line1)
        lang1.append(line1)

    i=1
    while True:
        if max_seq is not None:
            if len(lang1) >= max_seq:
                break
        line2 = file_handler2.readline()
        if not line2:
            break

        order2, line2 = line2.split('\t')
        order2 = int(order2.split('-')[1])
        if i == order2:
            i+=1
        else:
            print(order2, line2)
        lang2.append(line2)

    print(f'reading bucc {data_type}-set  {language}-en, total lines:', len(lang1), len(lang2))

    # Close
    file_handler1.close()
    file_handler2.close()

    return lang1, lang2


def write_bucc_gold_sents(language='zh', max_seq=None, data_type='training', line_lower_limit=10, random_sample=False, debug=True):
    lang1, lang2 = get_bucc_sentence(language, data_type=data_type) #index start from 0
    if data_type == 'training':
        file_handler = open(
            f"./dataset/bucc/bucc2018/{language}-en/{language}-en.training.gold", "r",
            encoding="utf-8")
    elif data_type == 'sample':
        file_handler = open(
            f"./dataset/bucc/bucc2018/{language}-en/{language}-en.sample.gold", "r",
            encoding="utf-8")

    with open(f'./dataset/bucc/bucc2018/{language}-en/{language}-en.{data_type}.{language}_gold_sents', 'w', encoding='utf-8') as f1:
        with open(f'./dataset/bucc/bucc2018/{language}-en/{language}-en.{data_type}.en_gold_sents', 'w', encoding='utf-8') as f2:
            while True:
                line = file_handler.readline()
                if not line:
                    break

                id1, id2 = line.split('\t')
                id1 = int(id1.split('-')[1])
                id2 = int(id2.split('-')[1])

                print(lang1[id1-1], file=f1, end='')
                print(lang2[id2-1], file=f2, end='')


def get_gold_sents(language='zh', max_seq=None, data_type='training', line_lower_limit=10, random_sample=False, debug=True):
    target_path_1 = f'./dataset/bucc/bucc2018/{language}-en/{language}-en.{data_type}.{language}_gold_sents'
    target_path_2 = f'./dataset/bucc/bucc2018/{language}-en/{language}-en.{data_type}.en_gold_sents'
    if not os.path.exists(target_path_1):
        write_bucc_gold_sents(language=language, data_type=data_type)

    lang1 = []
    lang2 = []
    f1 = open(target_path_1, encoding='utf-8')
    f2 = open(target_path_2, encoding='utf-8')
    while True:
        line1 = f1.readline()
        line2 = f2.readline()
        if not(line1 and line2):
            break
        else:
            lang1.append(line1)
            lang2.append(line2)

    return lang1, lang2


def suffle_sent(lang1, lang2):
    """Shuffle list x in place, and return None.
    原位打乱列表，不生成新的列表。

    Optional argument random is a 0-argument
    function returning a random float in [0.0, 1.0); 
    if it is the default None, 
    the standard random.random will be used.
	可选参数random是一个从0到参数的函数，返回[0.0,1.0)中的随机浮点；
	如果random是缺省值None，则将使用标准的random.random()。
    """
    for i in reversed(range(1, len(lang1))):
        # pick an element in x[:i+1] with which to exchange x[i]
        j = random.randint(0, i+1)
        lang1[i], lang1[j] = lang1[j], lang1[i]
        lang2[i], lang2[j] = lang2[j], lang2[i]
    return


def save_embedding_file(model_name, embeds_lang, embeds_en, data_type='training', language='zh', att=False):
    if att:
        print('saving to ->', end='')
        file_folder = f'./dataset/bucc/bucc2018/{language}-en/'

        embeds_file_lang = file_folder + f'{model_name}_{language}_{data_type}.npy'
        embeds_file_en = file_folder + f'{model_name}_en_{data_type}.npy'
        np.save(embeds_file_lang, embeds_lang)
        np.save(embeds_file_en, embeds_en)
        print(embeds_file_lang, embeds_file_en)
        return file_folder

    else:    
        assert isinstance(embeds_en, list)
        layer = len(embeds_en)
        save_layer = LAYER_LIST[layer]

        print('saving to ->', end='')
        file_folder = f'./dataset/bucc/bucc2018/{language}-en/'

        for i in save_layer:
            embeds_file_lang = file_folder + f'{model_name}_{language}_{data_type}_{i}.npy'
            embeds_file_en = file_folder + f'{model_name}_en_{data_type}_{i}.npy'


            np.save(embeds_file_lang, embeds_lang[i])
            np.save(embeds_file_en, embeds_en[i])
            print(embeds_file_lang, embeds_file_en)

        return file_folder


def load_bucc_embeddings(language, model_name, layer, data_type='training'):
    '''
    :param language:
    :param model_name:
    :param layer:
    :param data_type:
    :return: embedding_lang, embedding_en
    '''
    file_folder = f'./dataset/bucc/bucc2017/{language}-en/'

    if model_name == 'laser':
        if data_type == 'sample':
            file_folder = f'./dataset/bucc/bucc2018/{language}-en/'
            file_lang = file_folder + f'laser_{language}_{data_type}_{layer}.npy'
            file_en = file_folder + f'laser_en_{data_type}_{layer}.npy'
        else:
            file_lang = file_folder + f'laser_{language}_{layer}.npy'
            file_en = file_folder + f'laser_en_{layer}.npy'

        assert os.path.exists(file_lang) and os.path.exists(file_en), f'loading path {file_lang} {file_en} do not exist'
        lang = np.fromfile(file_lang, dtype=np.float32)
        en = np.fromfile(file_en, dtype=np.float32)
        return np.reshape(lang, (-1, 1024)), np.reshape(en, (-1, 1024))


    else: # other model
        if data_type == 'sample':
            file_folder = f'./dataset/bucc/bucc2018/{language}-en/'
            file_lang = file_folder + f'{model_name}_{language}_{data_type}_{layer}.npy'
            file_en = file_folder + f'{model_name}_en_{data_type}_{layer}.npy'
        else: # training
            file_lang = file_folder + f'{model_name}_{language}_{layer}.npy'
            file_en = file_folder + f'{model_name}_en_{layer}.npy'

        assert os.path.exists(file_lang) and os.path.exists(file_en), f'loading path {file_lang} {file_en} do not exist'
        print(f'load {file_lang}, {file_en}')
        return np.load(file_lang), np.load(file_en)


def load_att_embeddings(language, model_name, data_type='training'):
    file_folder = f'./dataset/bucc/bucc2018/{language}-en/'
    embeds_file_lang = file_folder + f'{model_name}_{language}_{data_type}.npy'
    embeds_file_en = file_folder + f'{model_name}_en_{data_type}.npy'
    return np.load(embeds_file_lang), np.load(embeds_file_en)


def filter_dup_embeddings(embeddings_lang, embeddings_en, gold):
    '''
        filter dup embeddings
        return raw index(0, ...) of filter embeddings
    '''
    lang_sent_id = np.arange(0, embeddings_lang.shape[0])
    en_sent_id = np.arange(0, embeddings_en.shape[0])
    dup_lang_sent = list(gold.dup_lang.keys())
    dup_en_sent = list(gold.dup_en.keys())

    np.delete(lang_sent_id, dup_lang_sent, axis=0)
    np.delete(embeddings_lang, dup_lang_sent, axis=0)

    np.delete(en_sent_id, dup_en_sent)
    np.delete(embeddings_en, dup_en_sent, axis=0)
    return lang_sent_id, en_sent_id



def remove_embedding_file(model_name, language='zh', data_type='training', layer=9):
    if data_type == 'sample':
        file_folder = f'./dataset/bucc/bucc2018/{language}-en/'
    else:
        file_folder = f'./dataset/bucc/bucc2017/{language}-en/'

    if data_type == 'sample':
        embeds_file_lang = file_folder + f'{model_name}_{language}_{data_type}_{layer}.npy'
        embeds_file_en = file_folder + f'{model_name}_en_{data_type}_{layer}.npy'
    else: # training
        embeds_file_lang = file_folder + f'{model_name}_{language}_{layer}.npy'
        embeds_file_en = file_folder + f'{model_name}_en_{layer}.npy'

    try:
        os.remove(embeds_file_en)
        os.remove(embeds_file_lang)
    except FileNotFoundError:
        print()
    else:
        print('remove', embeds_file_lang, embeds_file_en)

    return file_folder


if __name__ == '__main__':
    # for lang in ['zh']:
    #     write_bucc_gold_sents(language=lang, data_type='training')
    print(random.randint(0, 2))
