from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from draw_pic import reduce_to_2d_tsne, reduce_to_2d_pca, draw_embeddings
import os
# from sofa import VecoModel, VecoTokenizer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
debug = True
model_name = 'XLM-R-LARGE'
features_out_hook = []
# import sofa
# sofa.environ("huggingface")

def load_model():
    model = AutoModel.from_pretrained('./model/xlm-r-large/', )
    model.to(device)
    print(model)
    return model


def register_hook(pretrain_model):
    def hook(module, fea_in, fea_out):
        fea_out = torch.mean(fea_out[0], dim=1, keepdim=False)
        if fea_out.is_cuda:
            fea_out = fea_out.cpu()
            print('in cuda')
        fea_out = fea_out.numpy()
        features_out_hook.append(fea_out)
        # print(fea_out.shape)
        return None

    for layer in pretrain_model.encoder.layers:
        # print(model)
        layer.register_forward_hook(hook=hook)
        # if layer_name == layer_name:
        #     module.register_forward_hook(hook=hook)
        #     print('registered: ' + name)


def generate_embeddings(tokenizer, model):
    text1 = "杰克非常喜欢纽约！"
    text2 = "Jack loves New York very much!"

    input1 = tokenizer([text1, text2], return_tensors='pt', padding=True).to(device)

    # print(input1)
    with torch.no_grad():
        # 为什么使用**input
        # input是一个dict
        outputs = model(**input1)

    for i in range(len(features_out_hook)):
        features_out_hook[i] = reduce_to_2d_pca(features_out_hook[i])


def draw():
    embedding_layer = 3
    image = draw_embeddings(
        model_name=model_name,
        embeddings_2d=features_out_hook[embedding_layer],
        embedding_layer=embedding_layer,
        language=['en', 'zh'],
        counts=[1, 1],
        compression_method='pca',
        width=8000,
        height=8000,
        debug=debug
        )

    folder = './output/test/'
    # 生成文件名
    filename = model_name.replace('/', '_') + f'.{embedding_layer}.jpg'
    filename = 'embeddings_' + filename
    if folder is not None and len(folder) > 0:
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, filename)

    # save to file
    if debug:
        print(f"[{model_name}]: save {embedding_layer}_embeddings to {filename}...")
    image.save(filename, quality=80, optimize=True, progressive=True)


if __name__ == '__main__':
    model = load_model()
    print(model)

    # tokenizer, model = load_model()
    # register_hook(model)
    # generate_embeddings(tokenizer, model)
    # draw()
