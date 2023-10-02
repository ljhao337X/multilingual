import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

if __name__ =='__main__':
    # state_dict = torch.load('./model/laser/bilstm.93langs.2018-12-26.pt')
    # print(state_dict['params'])
    # {'num_embeddings': 73640, 'padding_idx': 1, 'embed_dim': 320, 'hidden_size': 512, 'num_layers': 5,
    #  'bidirectional': True, 'left_pad': True, 'padding_value': 0.0}
    tokenizer = AutoTokenizer.from_pretrained('./model/xlm-r/')
    print(tokenizer.tokenize('English translation'))
    # 不能对多个句子分词

    # x = np.load('./output/accuracy_INFOXLM/to_eng_to_afr.npy')
    # print(x.shape)


