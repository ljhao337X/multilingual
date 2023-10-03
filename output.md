# 模型结构

```
D:\program\anaconda\envs\nlp\python.exe E:\multilingual\encoder_test.py 
XLMRobertaConfig {
  "architectures": [
    "XLMRobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "xlm-roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "output_past": true,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.24.0",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 250002
}

Some weights of the model checkpoint at ./model/xlm-r/ were not used when initializing XLMRobertaModel: ['lm_head.decoder.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.bias']
- This IS expected if you are initializing XLMRobertaModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLMRobertaModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
XLMRobertaModel(
  (embeddings): XLMRobertaEmbeddings(
    (word_embeddings): Embedding(250002, 768, padding_idx=1)
    (position_embeddings): Embedding(514, 768, padding_idx=1)
    (token_type_embeddings): Embedding(1, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): XLMRobertaEncoder(
    (layer): ModuleList(
      (0): XLMRobertaLayer(
        (attention): XLMRobertaAttention(
          (self): XLMRobertaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): XLMRobertaSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): XLMRobertaIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): XLMRobertaOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (1): XLMRobertaLayer(
        (attention): XLMRobertaAttention(
          (self): XLMRobertaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): XLMRobertaSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): XLMRobertaIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): XLMRobertaOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (2): XLMRobertaLayer(
        (attention): XLMRobertaAttention(
          (self): XLMRobertaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): XLMRobertaSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): XLMRobertaIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): XLMRobertaOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (3): XLMRobertaLayer(
        (attention): XLMRobertaAttention(
          (self): XLMRobertaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): XLMRobertaSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): XLMRobertaIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): XLMRobertaOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (4): XLMRobertaLayer(
        (attention): XLMRobertaAttention(
          (self): XLMRobertaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): XLMRobertaSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): XLMRobertaIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): XLMRobertaOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (5): XLMRobertaLayer(
        (attention): XLMRobertaAttention(
          (self): XLMRobertaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): XLMRobertaSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): XLMRobertaIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): XLMRobertaOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (6): XLMRobertaLayer(
        (attention): XLMRobertaAttention(
          (self): XLMRobertaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): XLMRobertaSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): XLMRobertaIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): XLMRobertaOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (7): XLMRobertaLayer(
        (attention): XLMRobertaAttention(
          (self): XLMRobertaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): XLMRobertaSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): XLMRobertaIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): XLMRobertaOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (8): XLMRobertaLayer(
        (attention): XLMRobertaAttention(
          (self): XLMRobertaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): XLMRobertaSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): XLMRobertaIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): XLMRobertaOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (9): XLMRobertaLayer(
        (attention): XLMRobertaAttention(
          (self): XLMRobertaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): XLMRobertaSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): XLMRobertaIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): XLMRobertaOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (10): XLMRobertaLayer(
        (attention): XLMRobertaAttention(
          (self): XLMRobertaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): XLMRobertaSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): XLMRobertaIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): XLMRobertaOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (11): XLMRobertaLayer(
        (attention): XLMRobertaAttention(
          (self): XLMRobertaSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): XLMRobertaSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): XLMRobertaIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
        )
        (output): XLMRobertaOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): XLMRobertaPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)

embeddings
embeddings.word_embeddings
embeddings.position_embeddings
embeddings.token_type_embeddings
embeddings.LayerNorm
embeddings.dropout
encoder
encoder.layer
encoder.layer.0
encoder.layer.0.attention
encoder.layer.0.attention.self
encoder.layer.0.attention.self.query
encoder.layer.0.attention.self.key
encoder.layer.0.attention.self.value
encoder.layer.0.attention.self.dropout
encoder.layer.0.attention.output
encoder.layer.0.attention.output.dense
encoder.layer.0.attention.output.LayerNorm
encoder.layer.0.attention.output.dropout
encoder.layer.0.intermediate
encoder.layer.0.intermediate.dense
encoder.layer.0.intermediate.intermediate_act_fn
encoder.layer.0.output
encoder.layer.0.output.dense
encoder.layer.0.output.LayerNorm
encoder.layer.0.output.dropout
encoder.layer.1
encoder.layer.1.attention
encoder.layer.1.attention.self
encoder.layer.1.attention.self.query
encoder.layer.1.attention.self.key
encoder.layer.1.attention.self.value
encoder.layer.1.attention.self.dropout
encoder.layer.1.attention.output
encoder.layer.1.attention.output.dense
encoder.layer.1.attention.output.LayerNorm
encoder.layer.1.attention.output.dropout
encoder.layer.1.intermediate
encoder.layer.1.intermediate.dense
encoder.layer.1.intermediate.intermediate_act_fn
encoder.layer.1.output
encoder.layer.1.output.dense
encoder.layer.1.output.LayerNorm
encoder.layer.1.output.dropout
encoder.layer.2
encoder.layer.2.attention
encoder.layer.2.attention.self
encoder.layer.2.attention.self.query
encoder.layer.2.attention.self.key
encoder.layer.2.attention.self.value
encoder.layer.2.attention.self.dropout
encoder.layer.2.attention.output
encoder.layer.2.attention.output.dense
encoder.layer.2.attention.output.LayerNorm
encoder.layer.2.attention.output.dropout
encoder.layer.2.intermediate
encoder.layer.2.intermediate.dense
encoder.layer.2.intermediate.intermediate_act_fn
encoder.layer.2.output
encoder.layer.2.output.dense
encoder.layer.2.output.LayerNorm
encoder.layer.2.output.dropout
encoder.layer.3
encoder.layer.3.attention
encoder.layer.3.attention.self
encoder.layer.3.attention.self.query
encoder.layer.3.attention.self.key
encoder.layer.3.attention.self.value
encoder.layer.3.attention.self.dropout
encoder.layer.3.attention.output
encoder.layer.3.attention.output.dense
encoder.layer.3.attention.output.LayerNorm
encoder.layer.3.attention.output.dropout
encoder.layer.3.intermediate
encoder.layer.3.intermediate.dense
encoder.layer.3.intermediate.intermediate_act_fn
encoder.layer.3.output
encoder.layer.3.output.dense
encoder.layer.3.output.LayerNorm
encoder.layer.3.output.dropout
encoder.layer.4
encoder.layer.4.attention
encoder.layer.4.attention.self
encoder.layer.4.attention.self.query
encoder.layer.4.attention.self.key
encoder.layer.4.attention.self.value
encoder.layer.4.attention.self.dropout
encoder.layer.4.attention.output
encoder.layer.4.attention.output.dense
encoder.layer.4.attention.output.LayerNorm
encoder.layer.4.attention.output.dropout
encoder.layer.4.intermediate
encoder.layer.4.intermediate.dense
encoder.layer.4.intermediate.intermediate_act_fn
encoder.layer.4.output
encoder.layer.4.output.dense
encoder.layer.4.output.LayerNorm
encoder.layer.4.output.dropout
encoder.layer.5
encoder.layer.5.attention
encoder.layer.5.attention.self
encoder.layer.5.attention.self.query
encoder.layer.5.attention.self.key
encoder.layer.5.attention.self.value
encoder.layer.5.attention.self.dropout
encoder.layer.5.attention.output
encoder.layer.5.attention.output.dense
encoder.layer.5.attention.output.LayerNorm
encoder.layer.5.attention.output.dropout
encoder.layer.5.intermediate
encoder.layer.5.intermediate.dense
encoder.layer.5.intermediate.intermediate_act_fn
encoder.layer.5.output
encoder.layer.5.output.dense
encoder.layer.5.output.LayerNorm
encoder.layer.5.output.dropout
encoder.layer.6
encoder.layer.6.attention
encoder.layer.6.attention.self
encoder.layer.6.attention.self.query
encoder.layer.6.attention.self.key
encoder.layer.6.attention.self.value
encoder.layer.6.attention.self.dropout
encoder.layer.6.attention.output
encoder.layer.6.attention.output.dense
encoder.layer.6.attention.output.LayerNorm
encoder.layer.6.attention.output.dropout
encoder.layer.6.intermediate
encoder.layer.6.intermediate.dense
encoder.layer.6.intermediate.intermediate_act_fn
encoder.layer.6.output
encoder.layer.6.output.dense
encoder.layer.6.output.LayerNorm
encoder.layer.6.output.dropout
encoder.layer.7
encoder.layer.7.attention
encoder.layer.7.attention.self
encoder.layer.7.attention.self.query
encoder.layer.7.attention.self.key
encoder.layer.7.attention.self.value
encoder.layer.7.attention.self.dropout
encoder.layer.7.attention.output
encoder.layer.7.attention.output.dense
encoder.layer.7.attention.output.LayerNorm
encoder.layer.7.attention.output.dropout
encoder.layer.7.intermediate
encoder.layer.7.intermediate.dense
encoder.layer.7.intermediate.intermediate_act_fn
encoder.layer.7.output
encoder.layer.7.output.dense
encoder.layer.7.output.LayerNorm
encoder.layer.7.output.dropout
encoder.layer.8
encoder.layer.8.attention
encoder.layer.8.attention.self
encoder.layer.8.attention.self.query
encoder.layer.8.attention.self.key
encoder.layer.8.attention.self.value
encoder.layer.8.attention.self.dropout
encoder.layer.8.attention.output
encoder.layer.8.attention.output.dense
encoder.layer.8.attention.output.LayerNorm
encoder.layer.8.attention.output.dropout
encoder.layer.8.intermediate
encoder.layer.8.intermediate.dense
encoder.layer.8.intermediate.intermediate_act_fn
encoder.layer.8.output
encoder.layer.8.output.dense
encoder.layer.8.output.LayerNorm
encoder.layer.8.output.dropout
encoder.layer.9
encoder.layer.9.attention
encoder.layer.9.attention.self
encoder.layer.9.attention.self.query
encoder.layer.9.attention.self.key
encoder.layer.9.attention.self.value
encoder.layer.9.attention.self.dropout
encoder.layer.9.attention.output
encoder.layer.9.attention.output.dense
encoder.layer.9.attention.output.LayerNorm
encoder.layer.9.attention.output.dropout
encoder.layer.9.intermediate
encoder.layer.9.intermediate.dense
encoder.layer.9.intermediate.intermediate_act_fn
encoder.layer.9.output
encoder.layer.9.output.dense
encoder.layer.9.output.LayerNorm
encoder.layer.9.output.dropout
encoder.layer.10
encoder.layer.10.attention
encoder.layer.10.attention.self
encoder.layer.10.attention.self.query
encoder.layer.10.attention.self.key
encoder.layer.10.attention.self.value
encoder.layer.10.attention.self.dropout
encoder.layer.10.attention.output
encoder.layer.10.attention.output.dense
encoder.layer.10.attention.output.LayerNorm
encoder.layer.10.attention.output.dropout
encoder.layer.10.intermediate
encoder.layer.10.intermediate.dense
encoder.layer.10.intermediate.intermediate_act_fn
encoder.layer.10.output
encoder.layer.10.output.dense
encoder.layer.10.output.LayerNorm
encoder.layer.10.output.dropout
encoder.layer.11
encoder.layer.11.attention
encoder.layer.11.attention.self
encoder.layer.11.attention.self.query
encoder.layer.11.attention.self.key
encoder.layer.11.attention.self.value
encoder.layer.11.attention.self.dropout
encoder.layer.11.attention.output
encoder.layer.11.attention.output.dense
encoder.layer.11.attention.output.LayerNorm
encoder.layer.11.attention.output.dropout
encoder.layer.11.intermediate
encoder.layer.11.intermediate.dense
encoder.layer.11.intermediate.intermediate_act_fn
encoder.layer.11.output
encoder.layer.11.output.dense
encoder.layer.11.output.LayerNorm
encoder.layer.11.output.dropout
pooler
pooler.dense
pooler.activation

Process finished with exit code 0

```

# 输入
```python
text1 = "杰克非常喜欢纽约！"
text2 = "Jack loves New York very much!"
# {'input_ids': [0, 6, 32093, 3987, 4528, 14999, 82534, 38, 2], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

input1 = xlmr_tokenizer(text1, return_tensors='pt')
input2 = xlmr_tokenizer(text2, return_tensors='pt')
print(**input1)
with torch.no_grad():
    # 为什么使用**input
    # input是一个dict, 需要将其按key对应传参
    outputs = xlmr_model(**input)
```

# 输出
[1, 9, 729]

# dataset
tatoeba

language_reference:

https://tatoeba.org/en/stats/sentences_by_language

# accuracy
## INFOXLM
```shell
dismatch in to_eng_to_amh.npy
embedding got the best in to_eng_to_ang.npy
dismatch in to_eng_to_ast.npy
dismatch in to_eng_to_awa.npy
embedding got the best in to_eng_to_ber.npy
dismatch in to_eng_to_bre.npy
embedding got the best in to_eng_to_bre.npy
dismatch in to_eng_to_cha.npy
embedding got the best in to_eng_to_cha.npy
dismatch in to_eng_to_cor.npy
embedding got the best in to_eng_to_cor.npy
dismatch in to_eng_to_csb.npy
dismatch in to_eng_to_cym.npy
dismatch in to_eng_to_dtp.npy
embedding got the best in to_eng_to_dtp.npy
dismatch in to_eng_to_ell.npy
dismatch in to_eng_to_fra.npy
dismatch in to_eng_to_gla.npy
embedding got the best in to_eng_to_gla.npy
dismatch in to_eng_to_hye.npy
dismatch in to_eng_to_kab.npy
embedding got the best in to_eng_to_kab.npy
dismatch in to_eng_to_kzj.npy
embedding got the best in to_eng_to_kzj.npy
dismatch in to_eng_to_orv.npy
dismatch in to_eng_to_pam.npy
embedding got the best in to_eng_to_pam.npy
dismatch in to_eng_to_pms.npy
dismatch in to_eng_to_tam.npy
dismatch in to_eng_to_tat.npy
dismatch in to_eng_to_tzl.npy
embedding got the best in to_eng_to_tzl.npy
dismatch in to_eng_to_xho.npy
embedding got the best in to_eng_to_xho.npy
dismatch in to_eng_to_yid.npy
0.010529005523052451 [17, 1, 2, 0, 1, 7, 11, 4, 0, 3, 13, 32, 21] [12, 0, 0, 0, 0, 0, 0, 12, 87, 1, 0, 0, 0] [12, 0, 0, 0, 0, 0, 0, 12, 87, 1, 0, 0, 0]
languages where embedding 0 perform the best:  ['ang', 'ber', 'bre', 'cha', 'cor', 'dtp', 'gla', 'kab', 'kzj', 'pam', 'tzl', 'xho']
Process finished with exit code 0

```
## XLM-R
```shell
dismatch in to_eng_to_amh.npy
embedding got the best in to_eng_to_ang.npy
dismatch in to_eng_to_ast.npy
dismatch in to_eng_to_awa.npy
embedding got the best in to_eng_to_ber.npy
dismatch in to_eng_to_bre.npy
embedding got the best in to_eng_to_bre.npy
dismatch in to_eng_to_cha.npy
embedding got the best in to_eng_to_cha.npy
dismatch in to_eng_to_cor.npy
embedding got the best in to_eng_to_cor.npy
dismatch in to_eng_to_csb.npy
dismatch in to_eng_to_cym.npy
dismatch in to_eng_to_dtp.npy
embedding got the best in to_eng_to_dtp.npy
dismatch in to_eng_to_ell.npy
dismatch in to_eng_to_fra.npy
dismatch in to_eng_to_gla.npy
embedding got the best in to_eng_to_gla.npy
dismatch in to_eng_to_hye.npy
dismatch in to_eng_to_kab.npy
embedding got the best in to_eng_to_kab.npy
dismatch in to_eng_to_kzj.npy
embedding got the best in to_eng_to_kzj.npy
dismatch in to_eng_to_orv.npy
dismatch in to_eng_to_pam.npy
embedding got the best in to_eng_to_pam.npy
dismatch in to_eng_to_pms.npy
dismatch in to_eng_to_tam.npy
dismatch in to_eng_to_tat.npy
dismatch in to_eng_to_tzl.npy
embedding got the best in to_eng_to_tzl.npy
dismatch in to_eng_to_xho.npy
embedding got the best in to_eng_to_xho.npy
dismatch in to_eng_to_yid.npy
0.010529005523052451 [17, 1, 2, 0, 1, 7, 11, 4, 0, 3, 13, 32, 21] [12, 0, 0, 0, 0, 0, 0, 12, 87, 1, 0, 0, 0] [12, 0, 0, 0, 0, 0, 0, 12, 87, 1, 0, 0, 0]
languages where embedding 0 perform the best:  ['ang', 'ber', 'bre', 'cha', 'cor', 'dtp', 'gla', 'kab', 'kzj', 'pam', 'tzl', 'xho']
Process finished with exit code 0

```

## XLM-A
```shell
-0.01235226186541126 [26, 0, 0, 0, 2, 3, 3, 2, 6, 11, 6, 32, 21] [26, 0, 0, 0, 0, 0, 2, 5, 41, 38, 0, 0, 0] [26, 0, 0, 0, 0, 0, 2, 5, 41, 38, 0, 0, 0]
languages where embedding 0 perform the best:  ['ang', 'ber', 'bre', 'ceb', 'cha', 'cor', 'csb', 'dtp', 'fry', 'gla', 'gsw', 'ido', 'ile', 'jav', 'kab', 'kur', 'kzj', 'max', 'mhr', 'nds', 'nov', 'pam', 'pms', 'swg', 'tuk', 'tzl', 'war', 'xho'
```

# optimal layer
```shell
# XLM-R
layer8 have best avg acc of all language: 0.5284577297327658
optimal avg acc 0.5293364375421942
optimal layer count:  Counter({8: 104, 0: 8})
```


```shell
# INFOXLM bidirection-mean
layer8 have best avg acc of all language: 0.5284577297327658
optimal avg acc0.5293364375421942
optimal layer count: Counter({8: 104, 0: 8})

# en-x
layer8 have best avg acc of all language: 0.5479820752460932
optimal avg acc0.5484182426234918
Counter({8: 109, 0: 2, 3: 1})
```

```shell
# xlm-a bidirection-mean
layer8 have best avg acc of all language: 0.3412337165837456
optimal avg acc0.35517515860649607
optimal layer count:  Counter({8: 32, 0: 27, 9: 25, 7: 23, 6: 4, 3: 1})

```

```shell
# ernie-m bidirection-mean
layer9 have best avg acc of all language: 0.5326573644295222
optimal avg acc0.5395739512538532
optimal layer count:  Counter({9: 42, 7: 31, 8: 27, 0: 6, 6: 3, 2: 2, 1: 1})

```



# layerATT

所有语言training
只微调最后的attention层

```shell
# zh-training
optimized bucc f1: 0.0807825812559167, threshold: 0.07966890931129456, best_recall: 0.2469720905739863, best_precision: 0.10078740157480315
(0.0807825812559167, 0.2469720905739863)

# fr-sample
optimized bucc f1: 0.10968660968660969, threshold: 0.07685703039169312, best_recall: 0.39289558665231433, best_precision: 0.08195848855774349
(0.10968660968660969, 0.39289558665231433)

```