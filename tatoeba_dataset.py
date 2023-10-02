import os
import logging

import numpy as np

logger = logging.getLogger(__name__)
_text_dir = './dataset/tatoeba/v1/'
_embeds_dir = './dataset/embeds/'

def load_langs():
    langs = []
    for f in os.listdir(_text_dir):
        if f=='README.md':
            continue
        lang = f.split('.')[-1]
        if lang != 'eng':
            langs.append(lang)
    langs.sort()
    return langs


def load_sents(lang):
    '''
    :param lang:
    :return: tuple (txt_lang, txt_eng)
    '''
    file_lang = open(_text_dir+f"tatoeba.{lang}-eng.{lang}", encoding='utf-8')
    file_eng = open(_text_dir+f"tatoeba.{lang}-eng.eng", encoding='utf-8')

    logger.debug(f'loading {lang}-eng text...')

    txt_lang = []
    txt_eng = []

    while True:
        line_lang = file_lang.readline()
        line_eng = file_eng.readline()

        if line_lang and line_eng:
            txt_lang.append(line_lang)
            txt_eng.append(line_eng)
        else:
            return txt_lang, txt_eng

def load_embeds(model_name, lang, layer):
    file_lang = _embeds_dir+f"tatoeba.{lang}-eng.{lang}_{model_name}_{layer}.npy"
    file_eng = _embeds_dir+f"tatoeba.{lang}-eng.eng_{model_name}_{layer}.npy"

    return np.load(file_lang), np.load(file_eng)


def save_embeds(embed_lang, embed_eng, model_name, lang, layer):
    file_lang = _embeds_dir + f"tatoeba.{lang}-eng.{lang}_{model_name}_{layer}.npy"
    file_eng = _embeds_dir + f"tatoeba.{lang}-eng.eng_{model_name}_{layer}.npy"

    np.save(file_lang, embed_lang)
    np.save(file_eng, embed_eng)
    return



if __name__ == '__main__':
    print(load_langs())
