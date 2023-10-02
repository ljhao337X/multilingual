import torch
import numpy as np
import os
import logging
from tqdm import tqdm
from typing import List, AnyStr
from evaluate import load_model, MODEL_PATH, cls_pool_embedding, mean_pool_embedding
from bucc_dataset import get_bucc_sentence, save_embedding_file, load_bucc_embeddings, LAYER_LIST, Gold, filter_dup_embeddings
from draw_pic import draw_bucc_pic, reduce_to_2d_pca, reduce_to_2d_tsne
from evaluate_tools import generate_margin_k_similarity, bucc_optimize
import argparse
import matplotlib.pyplot as plt
# from bucc_laser import embed_sentences
BATCH_SIZE = 32
import logging
logger = logging.getLogger(__name__)

def run_inference(sentences, model_name, config, model, tokenizer, language='zh', pool_type='cls', debug=True):
    # config, model, tokenizer = load_model(model_path=MODEL_PATH[model_name], device=device, output_hidden_states=True)
    torch.cuda.empty_cache()
    sents_num = len(sentences)
    if debug:
        batch_num = 2
        sents_num = BATCH_SIZE * batch_num
    else:
        BATCH_SIZE = 256
        batch_num = -1 * (-sents_num // BATCH_SIZE)  # 向上除法
    embeds_size = config.hidden_size
    layer = config.num_hidden_layers + 1  # embedding layer
    logger.info(f'run inference sentence:{sents_num}, model:{model_name} has hidden_size{embeds_size}, and layers:{layer}')
    # batch_num = len(sentences_lang) / BATCH_SIZE
    all_embeds = [np.zeros(shape=(sents_num, embeds_size), dtype=np.float32) for _ in range(layer)]
    torch.cuda.empty_cache()
    for i in tqdm(range(batch_num), desc='Batch'):
        start_index = BATCH_SIZE * i
        end_index = min(sents_num, start_index + BATCH_SIZE)
        # batch：{input_ids, mask}
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