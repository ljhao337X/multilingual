import torch
import numpy as np
import os
import logging
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm
from bucc_dataset import get_bucc_sentence, save_embedding_file
# import faiss
from typing import List, AnyStr
from collections import Counter

logger = logging.getLogger(__name__)
def logger_init():
    logger.setLevel(level=logging.INFO)

    handler = logging.FileHandler("./log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    logger.addHandler(handler)

logger_init()

gpu = True
device = 'cuda:0' if torch.cuda.is_available() and gpu else 'cpu'
MODEL_NAME = 'INFOXLM'

EMBED_PATH = './dataset/embeds/'
TOKS_PATH = './dataset/toks/'
TEXT_PATH = './dataset/tatoeba/v1/'
MODEL_PATH = {'XLM-R':'./model/xlm-r/', 'INFOXLM': './model/infoxlm', 'ERNIE-M': './model/ernie-m/',
              'XLM-A':'./model/xlm-align', 'XLM-R-LARGE':'./model/xlm-r-large'}
BATCH_SIZE = 32
EMBED_SIZE = 768
LAYERS = 13

# 如果debug，则不会使用任何已写好的文件，也不会写入这些文件
DEBUG = False
SMALL_TEST = False
if DEBUG:
    EMBED_PATH = './dataset/embeds/test/'
    TOKS_PATH = './dataset/toks/test/'


def load_embeddings(embed_file, num_sentences=None):
    logger.info(' loading from {}'.format(embed_file))
    embeds = np.load(embed_file)
    return embeds

def load_model(model_path, device, output_hidden_states=None):
    '''
    :param model_path:
    :param output_hidden_states: set config.output_hidden_states=True
    :return: config, model, tokenizer
    '''
    config = AutoConfig.from_pretrained(model_path)
    if output_hidden_states is not None:
        config.output_hidden_states = output_hidden_states

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    logger.info("tokenizer.pad_token={}, pad_token_id={}".format(tokenizer.pad_token, tokenizer.pad_token_id))
    model = AutoModel.from_pretrained(model_path, config=config)

    # if args.init_checkpoint:
    #     model = model_class.from_pretrained(args.init_checkpoint, config=config, cache_dir=args.init_checkpoint)
    # else:
    #     model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(device)
    model.eval()
    return config, model, tokenizer


def prepare_batch(sentences, tokenizer, max_length=512, use_local_max_length=True, pool_skip_special_token=False, to_en=False, more_query=False):
    '''
    :param sentences: list of tokenized sentences
    :param tokenizer:
    :param max_length: pre-defined max_length
    :param use_local_max_length:
    :param pool_skip_special_token:
    :return: {"input_ids": input_ids -> list of [token_ids], "attention_mask": attention_mask, "token_type_ids": token_type_ids}, pool_mask,
    '''
    cls_token = tokenizer.cls_token
    if to_en:
        if more_query:
            cls_token = ['english', 'translation']
        else:
            cls_token = 'English'
        # refer to english
    sep_token = tokenizer.sep_token

    pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]

    batch_input_ids = []
    batch_attention_mask = []
    batch_pool_mask = []

    if more_query:
        local_max_length = min(max([len(s) for s in sentences]) + 3, max_length)
    else:
        local_max_length = min(max([len(s) for s in sentences]) + 2, max_length)
    if use_local_max_length:
        max_length = local_max_length

    for sent in sentences:
        # cut sent to max_length
        if len(sent) > max_length - 2:
            sent = sent[: (max_length - 2)]
        # add [cls]\[sep] and convert to ids
        if more_query and to_en:
            input_ids = tokenizer.convert_tokens_to_ids(cls_token + sent + [sep_token])
        else:
            input_ids = tokenizer.convert_tokens_to_ids([cls_token] + sent + [sep_token])

        padding_length = max_length - len(input_ids)
        # attention mask -> 【[cls] + sent + [sep]】 + paddings
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        # pool mask -> [cls] + 【sent】 + [sep] + paddings, used for retrieval
        pool_mask = [0] + [1] * (len(input_ids) - 2) + [0] * (padding_length + 1)
        # add padding tokens
        input_ids = input_ids + ([pad_token_id] * padding_length)

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_pool_mask.append(pool_mask)

    input_ids = torch.LongTensor(batch_input_ids).to(device)
    attention_mask = torch.LongTensor(batch_attention_mask).to(device)

    # get pool mask
    if pool_skip_special_token:
        pool_mask = torch.LongTensor(batch_pool_mask).to(device)
    else:
        pool_mask = attention_mask


def mean_pool_embedding(all_layer_outputs, masks):
    """
    Args:
      all_layer_outputs: list of embeds torch.FloatTensor, (B, L, D)
      masks: torch.FloatTensor, (B, L)
    Return:
      sent_emb: list of torch.FloatTensor, (B, D)
      remember use pool mask
    """
    sent_embeds = []
    for embeds in all_layer_outputs:
        embeds = (embeds * masks.unsqueeze(2).float()).sum(dim=1) / masks.sum(dim=1).view(-1, 1).float()
        sent_embeds.append(embeds)
    return sent_embeds


def cls_pool_embedding(all_layer_outputs):
    sent_embeds = []
    for embeds in all_layer_outputs:
        embeds = embeds[:, 0, :].squeeze(1)
        sent_embeds.append(embeds)
    return sent_embeds  # [B, D]


def run_inference(sentences:list, model_name, config, model, tokenizer:AutoTokenizer, pool_type='cls',
                  batch_size=BATCH_SIZE, to_en=False, debug=False):
    '''
    :param sentences:
    :param model_name:
    :param config:
    :param model:
    :param tokenizer:
    :param language:
    :param pool_type:
    :param batch_size:
    :param debug:
    :return: list of embeds: list of npArray[sents, hidden_size]
    '''
    # print(sentences)
    torch.cuda.empty_cache()
    sents_num = len(sentences)
    batch_num = -1 * (-sents_num // batch_size)  # 向上除法
    if debug:
        batch_num = 2
        sents_num = batch_num*batch_size
    layer = config.num_hidden_layers + 1
    embeds_size = config.hidden_size

    all_embeds = [np.zeros(shape=(sents_num, embeds_size), dtype=np.float32) for _ in range(layer)]
    if batch_num >= 10:
        # use tqdm
        logger.info(f'run inference sentence:{sents_num}, model:{model_name} has hidden_size{embeds_size}, and layers:{layer}')
        for i in tqdm(range(batch_num), desc='Batch'):
            start_index = batch_size * i
            end_index = min(sents_num, start_index + batch_size)
            # tokenized_sentence = [tokenizer.tokenize(x) for x in sentences[start_index:end_index]]
            # batch, pool_mask = prepare_batch(tokenized_sentence, tokenizer, to_en=to_en)
            batch = tokenizer(sentences[start_index: end_index], padding=True, return_tensors='pt').to(device)

            with torch.no_grad():
                outputs = model(**batch)[2]
                if pool_type == 'cls':
                    # [layer, batch, d]
                    batch_embeds_layers = cls_pool_embedding(outputs[-layer:])
                else:
                    batch_embeds_layers = mean_pool_embedding(outputs, batch['attention_mask'])

                for embeds, batch_embeds in zip(all_embeds, batch_embeds_layers):
                    embeds[start_index:end_index] = batch_embeds.cpu().numpy().astype(np.float32)

                del outputs
                torch.cuda.empty_cache()
    
    else:
        print('inferrence....')
        for i in range(batch_num):
            start_index = batch_size * i
            end_index = min(sents_num, start_index + batch_size)
            batch = tokenizer(sentences[start_index: end_index], padding=True, return_tensors='pt').to(device)

            with torch.no_grad():
                outputs = model(**batch)[2]
                if pool_type == 'cls':
                    # [layer, batch, d]
                    batch_embeds_layers = cls_pool_embedding(outputs[-layer:])
                else:
                    batch_embeds_layers = mean_pool_embedding(outputs, batch['attention_mask'])

                for embeds, batch_embeds in zip(all_embeds, batch_embeds_layers):
                    embeds[start_index:end_index] = batch_embeds.cpu().numpy().astype(np.float32)

                del outputs
                torch.cuda.empty_cache()

    # torch.cuda.empty_cache()
    return all_embeds


def extract_embeddings_save(model_name='XLM-R', languages=['zh'], pool_type='cls', data_type='training', debug=True):
    config, model, tokenizer = load_model(model_path=MODEL_PATH[model_name], device=device, output_hidden_states=True)

    for language in languages:
        sentences_lang, sentences_en = get_bucc_sentence(language=language, data_type=data_type)
        logger.info(f'read bucc {data_type} file {language}-en, total lines:{len(sentences_lang), len(sentences_en)}')
        all_embeds_lang = run_inference(sentences_lang, model_name, config, model, tokenizer, language,
                                        pool_type=pool_type, debug=debug)
        all_embeds_en = run_inference(sentences_en, model_name, config, model, tokenizer, 'en', pool_type=pool_type,
                                      debug=debug)

        file_folder = save_embedding_file(model_name=model_name, embeds_lang=all_embeds_lang, embeds_en=all_embeds_en,
                                          data_type=data_type, language=language)
        del all_embeds_lang, all_embeds_en

        logger.info(f'saved file to {file_folder}')
    return


if __name__ == '__main__':
    pass