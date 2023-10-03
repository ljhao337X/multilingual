import math
import time

from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import numpy as np
import os
import logging
from tqdm import tqdm
logger = logging.getLogger("__main__")
MODEL_PATH = {'XLM-R':'./model/xlm-r/', 'INFOXLM': './model/infoxlm', 'ERNIE-M': './model/ernie-m/',
              'XLM-A':'./model/xlm-align', 'XLM-R-LARGE':'./model/xlm-r-large'}
gpu = True
DEVICE = 'cuda:0' if torch.cuda.is_available() and gpu else 'cpu'
from bucc_dataset import get_bucc_sentence, get_gold_sents,save_embedding_file, load_att_embeddings, Gold, suffle_sent
# from bucc_evaluate import extract_embedding_att_save, filter_dup_embeddings, restore_index
# from evaluate_tools import generate_margin_k_similarity, bucc_optimize
from utils import accuracy, AverageMeter

def load_pretrain_model(model_path, device, output_hidden_states=None):
    '''
    :param model_path:
    :param output_hidden_states: set config.output_hidden_states=True
    :return: config, model(eval), tokenizer
    '''
    config = AutoConfig.from_pretrained(model_path)
    if output_hidden_states is not None:
        config.output_hidden_states = output_hidden_states

    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    # logger.info("tokenizer.pad_token={}, pad_token_id={}".format(tokenizer.pad_token, tokenizer.pad_token_id))
    model = AutoModel.from_pretrained(model_path, config=config)
    return config, model

def mean_pool_embedding(all_layer_outputs, masks) -> torch.Tensor:
    # [L, B, D]
    sent_embeds = []
    for embeds in all_layer_outputs:
        embeds = (embeds * masks.unsqueeze(2).float()).sum(dim=1) / masks.sum(dim=1).view(-1, 1).float()
        sent_embeds.append(embeds)
    return torch.stack(sent_embeds).permute(1, 0, 2) # [B, L, D]


def cls_pool_embedding(all_layer_outputs) -> torch.Tensor:
    sent_embeds = []
    for embeds in all_layer_outputs:
        embeds = embeds[:, 0, :].squeeze(1)
        sent_embeds.append(embeds)
    # [L, B, D]
    return torch.stack(sent_embeds).permute(1, 0, 2)  # [B, L, D]


class LAYER_ATT(nn.Module):
    def __init__(self, pretrain_model_name, device=DEVICE):
        super(LAYER_ATT, self).__init__()
        self.backbone_name = pretrain_model_name
        self.config, self.backbone = load_pretrain_model(
            MODEL_PATH[pretrain_model_name],
            device=device,
            output_hidden_states=True)
        self.backbone.eval()
        self.key = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False),
            nn.RReLU()
        )

        self.query = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False),
            nn.RReLU()
        )

        self.value = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False),
            nn.RReLU()
        )
        self.device = device
        self.backbone_path = MODEL_PATH[pretrain_model_name]


    def forward(self, batch):
        outputs = self.backbone(**batch)
        # 这里的output由于设置了output_hidden_states = true 所以包含三项
        all_layer_outputs = outputs[2]
        # [B, L, D]
        # still on device
        cls_embeddings = cls_pool_embedding(all_layer_outputs[-(self.config.num_hidden_layers+1):])
        avg_embeddings = mean_pool_embedding(all_layer_outputs, batch['attention_mask'])

        q = self.query(cls_embeddings)  # [bs, L, hid_size]
        k = self.key(avg_embeddings)  # [bs, L, hid_size]
        v = self.value(avg_embeddings)  # [bs, L, hid_size]

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.config.hidden_size)  # [bs, L, L]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, L, L]

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # 矩阵相乘，[bs, L, L]*[bs, L, 16] = [bs, L, 16]
        context_layer = torch.matmul(attention_probs, v)  # [bs, L, 16]
        return torch.mean(context_layer, dim=1)



class Trainer:
    def __init__(self, model, tokenizer, batch_size=32, lr=0.001, device=DEVICE):
        self.model_name = model.backbone_name+'_layer_att'
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        print(f"lr: {lr}, batch_size: {batch_size}")


    def train(self, langs, epoch=5):
        torch.cuda.empty_cache()
        lang1_sents, lang2_sents = [], []
        for lang in langs:
            t1, t2 = get_gold_sents(lang)
            lang1_sents += t1
            lang2_sents += t2
        suffle_sent(lang1_sents, lang2_sents)
        sent_num = len(lang1_sents)
        assert len(lang2_sents) == sent_num, print("gold sents dont equal")

        batch_num = -1 * (-sent_num // self.batch_size)  # 向上除法
        layer = self.model.config.num_hidden_layers + 1
        embeds_size = self.model.config.hidden_size

        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        for e in range(epoch):
            start = time.time()
            best_acc = 90

            for i in tqdm(range(batch_num), desc='Batch'):
                start_index = self.batch_size * i
                end_index = min(sent_num, start_index + self.batch_size)

                batch1 = self.tokenizer(lang1_sents[start_index:end_index], padding=True, return_tensors='pt').to(self.device)
                query = self.model(batch1)
                batch2 = self.tokenizer(lang2_sents[start_index:end_index], padding=True, return_tensors='pt').to(self.device)
                key = self.model(batch2) # [B, D]

                logits = query @ key.t() # [B, B]
                labels = torch.arange(logits.shape[0], device=DEVICE)
                loss = self.loss_fn(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
                losses.update(loss.item(), logits.size(0))
                top1.update(acc1[0], logits.size(0))
                top5.update(acc5[0], logits.size(0))

            batch_time.update(time.time() - start)
            if top1.avg > best_acc:
                time_mark = time.strftime("%m-%d-%H-%M-%S")
                model_path = f'./model_{time_mark}.pt'
                torch.save(self.model, f'./model_{time_mark}.pt')
                logger.info(f'save model {model_path} ... ')

            logger.info(f"Epoch:{e + 1}: {batch_time}, {losses}, {top1}, {top5}")

        return model_path


def att_inference(sents:list, model, tokenizer, batch_size=256, device=DEVICE):
    sents_num = len(sents)
    batch_num = -1 * (-sents_num // batch_size)  # 向上除法

    res = []
    with torch.no_grad():  # this is important
        for i in tqdm(range(batch_num), desc='Batch'):
            start_index = i*batch_size
            end_index = min(start_index+batch_size, sents_num)
            batch = tokenizer(sents[start_index:end_index], padding=True, return_tensors='pt').to(device)
            embeds = model(batch) # [sent, batch]
            res.append(embeds.cpu().numpy().astype(np.float32))

    if len(res) == 1:
        return res[0]
    return np.vstack(res)


def att_extract_save(model, tokenizer, languages=['fr'], data_type='training', device=DEVICE):
    model = model.to(device)
    model.eval()
    for language in languages:
        sentences_lang, sentences_en = get_bucc_sentence(language=language, data_type=data_type)
        logger.info(f'read bucc {data_type} file {language}-en, total lines:{len(sentences_lang), len(sentences_en)}')

        embeds_lang = att_inference(sentences_lang, model=model, tokenizer=tokenizer, device=device)
        embeds_en = att_inference(sentences_en, model=model, tokenizer=tokenizer, device=device)

        print(f'embeds_{language}: {embeds_lang.shape}, embeds_en: {embeds_en.shape}')
        file_folder = save_embedding_file("att"+model.backbone_name, embeds_lang, embeds_en, data_type=data_type, language=language, att=True)
        del embeds_lang, embeds_en

        logger.info(f'saved file to {file_folder}')
    return

if __name__ == '__main__':
    backbone_model = 'XLM-R'
    # # training
    # model = LAYER_ATT(backbone_model) # remember this
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[backbone_model])
    # trainer = Trainer(model, tokenizer)
    # model_path = trainer.train(['zh', 'fr', 'de', 'ru'], epoch=30)

    model = torch.load('./model_10-02-23-12-50.pt')
    tokenizer = AutoTokenizer.from_pretrained(model.backbone_path)
    att_extract_save(model, tokenizer, data_type='sample')
    

    # embeddings_lang, embeddings_en = load_att_embeddings(language='zh', model_name="att"+backbone_model, data_type='sample')
    # gold = Gold(language='zh', data_type='sample')
    # lang_sent_id, en_sent_id = filter_dup_embeddings(embeddings_lang, embeddings_en, gold)
    # candidate_and_score = generate_margin_k_similarity(embeddings_lang, embeddings_en, cosine=True, margin=True)

    # restore_index(lang_sent_id, en_sent_id, candidate_and_score)
    # best_f1, threshold, best_recall, best_precision = bucc_optimize(candidate_and_score, gold, ngold=gold.count)
    # logger.info(f'optimized bucc f1: {best_f1}, threshold: {threshold}, best_f1: {best_recall}, best_precision: {best_precision}')

    






