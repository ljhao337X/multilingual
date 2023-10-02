from transformers import AutoTokenizer
from transformers import XLMRobertaTokenizer

def get_dataset():
    fileHandler1 = open("./dataset/opus-100-corpus-zeroshot-v1.0/opus-100-corpus/v1.0/zero-shot/fr-zh/opus.fr-zh-test.zh", "r", encoding = "utf-8")
    fileHandler2 = open("./dataset/opus-100-corpus-zeroshot-v1.0/opus-100-corpus/v1.0/zero-shot/fr-zh/opus.fr-zh-test.fr", "r", encoding = "utf-8")
    lang1 = []
    lang2 = []

    while True:
        # Get next line from file
        line1 = fileHandler1.readline()
        line2 = fileHandler2.readline()

        # If line is empty then end of file reached
        if not line1 or not line2:
            break

        if len(line1) < 5 or len(line2) < 5:
            pass
        else:
            lang1.append(line1)
            lang2.append(line2)

        if len(lang1) % 500 == 0:
            print("read lines: ", len(lang1))
    # Close
    fileHandler1.close()
    fileHandler2.close()
    return lang1, lang2


# bert_model_name = "bert-base-cased"
# xlmr_model_name = "xlm-roberta-base"
# bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
# xlmr_tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)
xlmr_tokenizer = XLMRobertaTokenizer.from_pretrained('./model/xlm-r')

model_name = 'XLM-R'
text1 = "Jack Sparrow loves New York!"
text2 = "杰克非常喜欢纽约！"
input = [text1, text2]
print(xlmr_tokenizer.tokenize(text1))
print(xlmr_tokenizer.tokenize(text2))
print(xlmr_tokenizer(input, padding='longest', max_length=40))

# xlmr_tokens = xlmr_tokenizer.tokenize(text)
# xlmr_word_embeddings = xlmr_tokens.


# x, y = get_dataset()
# x_size = len(x)
# print("zh size:", x_size)
#
# xlmr_input = xlmr_tokenizer(x+y)
# print(xlmr_input)
# xlmr_tokens = xlmr_tokenizer(text).tokens()


def get_vocab(model_name: str, tokenizer, debug=False):
    # if "OpenAI" in model_name:
    #     model_name = model_name.split("/")[-1]
    #     return get_vocab_openai(model_name, debug=debug)
    #
    vocab_size = max([id for id in tokenizer.get_vocab().values()]) + 1
    if debug:
        print(f"[{model_name}]: vocab_size: {vocab_size}")

    # get vocab
    vocab = [''] * (vocab_size)
    for k, v in tokenizer.get_vocab().items():
        if v >= vocab_size:
            print(f"[{model_name}] out of range: {k}, {v}")
            continue
        try:
            if hasattr(tokenizer, 'convert_tokens_to_string'):
                vocab[v] = tokenizer.convert_tokens_to_string([k])
            # elif hasattr(tokenizer, 'text_tokenizer') and hasattr(tokenizer.text_tokenizer,
            #                                                       'convert_tokens_to_string'):
            #     # BAAI/aquila-7b
            #     vocab[v] = tokenizer.text_tokenizer.convert_tokens_to_string([k])
            # else:
            #     vocab[v] = k
        except Exception as e:
            print(f"[{model_name}]: convert_tokens_to_string({k}) failed: {e}")
            vocab[v] = k
    # if hasattr(tokenizer, 'convert_tokens_to_string'):
    #     print('has convert')
    if debug:
        print(vocab[10:100])
    return vocab


# get_vocab(model_name, xlmr_tokenizer, True)