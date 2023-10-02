import numpy
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
from draw_pic import reduce_to_2d_tsne, reduce_to_2d_pca, draw_embeddings, reduce_to_2d_mds, draw_tatoeba_pic
import os
import numpy as np
from evaluate import load_embeddings, tatoeba_retrieval_evaluate

DEBUG = True
cuda = True
device = "cuda:0" if torch.cuda.is_available() and cuda else "cpu"
size = 1024*4  # 读入句子的数量（一种语言） 1024*4
batch_size = 32
tsne_enable = True
language = ['zh', 'en']
# modelName = 'XLM-R'
modelName = 'INFOXLM'
# modelName = 'ERNIE-M'
MODEL_PATH = './model/infoxlm/'

# 保存中间结果
save = True
# 请修改语言类型、对应的模型文件路径


def get_dataset(max_length=None, line_lower_limit=10, debug=False):
    fileHandler1 = open(
        "./dataset/en-zh_CN.txt/CCAligned.en-zh_CN.zh_CN", "r",
        encoding="utf-8")
    fileHandler2 = open(
        "./dataset/en-zh_CN.txt/CCAligned.en-zh_CN.en", "r",
        encoding="utf-8")
    # fileHandler1 = open(
    #     "./dataset/opus-100-corpus-zeroshot-v1.0/opus-100-corpus/v1.0/zero-shot/fr-zh/opus.fr-zh-test.zh", "r",
    #     encoding="utf-8")
    # fileHandler2 = open(
    #     "./dataset/opus-100-corpus-zeroshot-v1.0/opus-100-corpus/v1.0/zero-shot/fr-zh/opus.fr-zh-test.fr", "r",
    #     encoding="utf-8")
    lang1 = []
    lang2 = []

    while True:
        if max_length is not None:
            if len(lang1) >= max_length:
                break

        # Get next line from file
        line1 = fileHandler1.readline()
        line2 = fileHandler2.readline()

        # If line is empty then end of file reached
        if not line1 or not line2:
            break

        if len(line1) < line_lower_limit or len(line2) < line_lower_limit:
            pass
        elif np.random.randint(0, 100) >= 50:
            pass
        else:
            lang1.append(line1)
            lang2.append(line2)
        if debug and len(lang1) % 500 == 0:
            print("read lines: ", len(lang1))

    # Close
    fileHandler1.close()
    fileHandler2.close()

    return lang1, lang2


def get_embedding_output(model_name, compression_method, layer_outputs, language_list):
    for x in range(len(layer_outputs)):
        if compression_method == 'PCA':
            embeddings = reduce_to_2d_pca(layer_outputs[x])
        elif compression_method == 'TSNE' and tsne_enable:
            embeddings = reduce_to_2d_tsne(layer_outputs[x])
        elif compression_method == 'MSD':
            try:
                embeddings = reduce_to_2d_mds(layer_outputs[x])
            except Exception as error:
                print(error)
                print('catch error when applying MDS')
                continue
        else:
            return

        # print(len(embeddings), embeddings[0].shape)
        save_image(model_name, compression_method, embeddings, language_list, layer=x)
        del embeddings


def save_image(model_name, compression_method, embeddings_2d, language_list, layer, debug=False):
    if debug:
        print(model_name, compression_method, embeddings_2d.shape)
    image = draw_embeddings(
        model_name=model_name,
        embeddings_2d=embeddings_2d,
        compression_method=compression_method,
        embedding_layer=layer,
        language=language_list,
        counts=[len(embeddings_2d)/2, len(embeddings_2d)/2],
        width=8000,
        height=8000,
        debug=debug
    )

    folder = './output/en-zh/'
    if debug:
        folder += 'test/'
    # 生成文件名
    filename = model_name.replace('/', '_') + '_' + compression_method + f'.{layer}.jpg'
    filename = 'embeddings_' + filename
    if folder is not None and len(folder) > 0:
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, filename)

    # save to file
    print(f"[{model_name}]: save {layer}_embeddings to {filename}...")
    image.save(filename, quality=30, optimize=True, progressive=True)
    del image


def save_embeddings(output_embeddings, model_name):
    import pickle

    path = './tmp' + '_' + model_name + '.pkl'

    f = open(path, 'wb')
    content = pickle.dumps(output_embeddings)
    f.write(content)
    f.close()


def pooling(all_layer_outputs, masks):
    '''
    :param all_layer_outputs: list of Tensor[B, L, E]
    :param masks: [B, L]
    :return: list of numpy [B, E]
    '''
    pooled_embeds = []
    for embeds in all_layer_outputs:
        # [B, L, E] * [B, L, 1] / [B, L]
        # [B, E] / [B, 1]
        embeds = (embeds * masks.unsqueeze(2).float()).sum(dim=1) / masks.sum(dim=1).view(-1, 1).float()
        pooled_embeds.append(embeds.cpu().numpy())
    return pooled_embeds

# def integrate_data(data_list:list[list[float]], layers:int):
#     for i in range(len(layers)):
#         data_list[i] = np.concatenate((data_list[x] for x in range(i, len(data_list), layers)))
#     return


def main(model_path, debug=False):
    if cuda:
        print("cleaning cache")
        torch.cuda.empty_cache()

    config = AutoConfig.from_pretrained(model_path)
    config.output_hidden_states = True
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    xlm_model = AutoModel.from_pretrained(model_path, config=config)
    if cuda:
        xlm_model = xlm_model.to(device)
    xlm_model.eval()

    # register hook
    features_out_hook = []
    layers = 13
    # cur = 0
    # def hook(module, fea_in, fea_out):
    #     fea_out = torch.mean(fea_out[0], dim=1, keepdim=False)
    #     if fea_out.is_cuda:
    #         fea_out = fea_out.cpu()
    #     fea_out = fea_out.numpy()
    #     features_out_hook.append(fea_out)
    #     return None
    #
    # if modelName == 'ERNIE-M':
    #     for model in xlm_model.encoder.layers:
    #         # print(model)
    #         model.register_forward_hook(hook=hook)
    #         layers += 1
    # else:
    #     for model in xlm_model.encoder.layer:
    #         # print(model)
    #         model.register_forward_hook(hook=hook)
    #         layers += 1

    # get embeddings
    x, y = get_dataset(max_length=size)
    counts = [len(x), len(y)]
    if debug:
        print(counts)
    input = x+y

    for i in range(0, len(input), batch_size):
        batch = tokenizer(input[i:i + batch_size], return_tensors='pt', padding='longest')
        if cuda:
            batch = batch.to(device)
        with torch.no_grad():
            # 为什么使用**input
            # input是一个dict
            outputs = xlm_model(**batch)
            # print(outputs)
            if cuda:
                sent_embeds = pooling(outputs[2], batch['attention_mask'])
                features_out_hook += sent_embeds
                # print([x.shape for x in sent_embeds])
            else:
                features_out_hook += [x.numpy() for x in outputs[2]]
        # print('predicted a batch:', i)
        del batch
        torch.cuda.empty_cache()
    del tokenizer
    del xlm_model
    del input

    # integrate data
    if len(features_out_hook) > layers:
        for i in range(len(features_out_hook)-1, layers-1, -1):
            features_out_hook[i%layers] = np.concatenate((features_out_hook[i%layers], features_out_hook[i]))
            del features_out_hook[i]
    if debug:
        print(len(features_out_hook), features_out_hook[0].shape)

    if save:
        save_embeddings(features_out_hook, modelName)

    compression = 'PCA'
    get_embedding_output(model_name=modelName, compression_method=compression, layer_outputs=features_out_hook,
                         language_list=language)

    compression = 'TSNE'
    get_embedding_output(model_name=modelName, compression_method=compression, layer_outputs=features_out_hook,
                         language_list=language)


def tatoeba_pic(model_name, layer, languages, compression_method, debug=False):
    embed_path = './dataset/embeds/'
    embeds = []
    for lang1 in languages:
        # an embed_file name should be "{text_file}_{model_name}_{layer}.npy"
        embed_file_lang1 = f'tatoeba.{lang1}-eng.{lang1}_{model_name}_{layer}.npy'
        embed_file_eng = f'tatoeba.{lang1}-eng.eng_{model_name}_{layer}.npy'
        print(f'loading from {embed_file_lang1}, {embed_file_eng}')
        embed_lang1 = np.load(embed_path + embed_file_lang1)
        embed_eng = np.load(embed_path + embed_file_eng)
        # print(embed_lang1, embed_eng)
        embeds += [embed_lang1, embed_eng]

    embeds = np.concatenate(embeds)
    print(embeds.shape)
    if compression_method=='pca':
        embed_s = reduce_to_2d_pca(embeds, debug=debug)
    elif compression_method=='tsne':
        embed_s = reduce_to_2d_tsne(embeds, debug=debug)

    image = draw_tatoeba_pic(model_name, embed_s, languages, compression_method, embedding_layer=layer, debug=True)

    folder = './output/tatoeba/'
    if debug:
        folder += 'test/'
    # 生成文件名
    filename = f'tatoeba_{model_name}_{compression_method}_{layer}.jpg'
    if folder is not None and len(folder) > 0:
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, filename)

    # save to file
    print(f"[{model_name}]: save {layer}_embeddings to {filename}...")
    image.save(filename, quality=50, optimize=True, progressive=True)
    del image


if __name__ == '__main__':
    TEXT_PATH = './dataset/tatoeba/v1/'
    MODEL_NAME = 'XLM-A'
    # 计算embeddings 和 accuracy
    tatoeba_files = os.listdir(path=TEXT_PATH)
    for file in tatoeba_files:
        if 'README' in file:
            continue
        # get language name
        lang1 = file.split('.')[-1]
        if lang1 == 'eng':
            continue
        # logger.info('calculating accuracy in {}'.format(lang1))
        # get output
        to_eng, to_lang1 = tatoeba_retrieval_evaluate(lang1, layers=[i for i in range(13)], model_name=MODEL_NAME, k=5)
        file = './output/accuracy_{}/to_eng_to_{}.npy'.format(MODEL_NAME, lang1)

        np.save(file, np.concatenate((to_eng, to_lang1)))

    # main(model_path=MODEL_PATH, debug=DEBUG)
    for i in range(13):
        tatoeba_pic(MODEL_NAME, i, languages=['fra', 'cmn', 'ind', 'jpn', 'rus'], compression_method='pca')
    for i in range(13):
        tatoeba_pic(MODEL_NAME, i, languages=['fra', 'cmn', 'ind', 'jpn', 'rus'], compression_method='tsne')
    # has = {'zh': '#B04759', 'en': '#FFFF00', 'ru': '#F99B7D', 'jp': '#146C94', 'id': '#19A7CE', 'fr': '#0000CC'}






