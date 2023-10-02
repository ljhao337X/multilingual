from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import random
from typing import List
from PIL import Image, ImageDraw, ImageColor, ImageFont
from sklearn.preprocessing import MinMaxScaler
from evaluate import lang3_dict

lanuage_color_map = {'zh':'#FF0000', 'en':'#0000FF', 'ru':'#808080', 'jp':'#FF00FF', 'id':'#FFFF00', 'fr':'#008000', 'de':'#FF00FF'}
bucc_color_map = {'zh':'#FF0000', 'en':'#0000FF', 'ru':'#808080', 'de':'#FFFF00', 'fr':'#008000'}

def get_language_color(language:str):
    '''
    :param language:
    :return: 语言对应的颜色
    '''
    return lanuage_color_map[language]


def reduce_to_2d_mds(embeddings, debug=False):
    '''
    尽量保持距离
    '''
    mds_model = MDS(n_components=2, n_jobs=-1)
    res = mds_model.fit_transform(embeddings)
    del mds_model
    return res

def reduce_to_2d_pca(embeddings, debug=False):
    pca_model = PCA(n_components=2)
    res = pca_model.fit_transform(embeddings)
    del pca_model
    return res


def reduce_to_2d_tsne(embeddings, debug=False):
    from sklearn.manifold import TSNE
    tsne_model = TSNE(n_components=2,
                      early_exaggeration=12,
                      # metric='cosine',
                      init='pca',
                      verbose=2 if debug else 0,
                      n_iter=1000,
                      random_state=42,
                      n_jobs=-1)

    # a new tsne based on cuda
    # from tsnecuda import TSNE
    # tsne_model = TSNE(n_components=2, perplexity=15, learning_rate=10)
    embeddings_2d = tsne_model.fit_transform(embeddings)
    del tsne_model

    return embeddings_2d

default_palette = [
    '#B04759', '#E76161', '#F99B7D',
    '#146C94', '#19A7CE', '#E893CF',
]


# apt install fonts-noto-cjk fonts-anonymous-pro fonts-noto-color-emoji
def get_available_font_from_list(fonts: List[str], size=14):
    for font in fonts:
        try:
            # 尝试加载字体文件
            # 如果加载成功，跳出循环
            return ImageFont.truetype(font, size=size)
        except IOError:
            print('fail loading chinese font')
            continue  # 如果加载失败，继续尝试下一个字体
    return None


# def get_chinese_font(size=14):
#     font_paths = [
#         "NotoSansMonoCJKsc-Regular",
#         "NotoSansCJKsc-Regular",
#         "NotoSansCJK",
#         "NotoSansCJK-Regular",
#         "NotoSerifCJK",
#         "NotoSerifCJK-Regular",
#         "STHeiti Light",
#         "微软雅黑",
#     ]
#     return get_available_font_from_list(font_paths, size=size)


def get_english_font(size=14):
    font_paths = [
        "Anonymous Pro",
        "DejaVuSansMono",
        "Arial",
    ]
    font = ImageFont.load_default()
    return font

def get_font(size=14):
    try:
        font = ImageFont.truetype(r'C:\Windows\Fonts\simkai.ttf', size=size)
    except OSError as e:
        font = ImageFont.truetype('./dataset/font/simkai.ttf', size=size)
    else:
        pass

    return font

import math


# margin = 0
# image_width = 0
# image_height = 0
# width = 0
# height = 0
# font_size = 0
# zh_font = None
def calculate_size(width, height, points_num, min_font_size=32):
    '''

    :param width:
    :param height:
    :param points_num:
    :param min_font_size:
    :return: margin, image_width, image_height, font_size, legend_size
    '''
    margin = int(width / 20)
    image_width = width + margin * (2 + 2)  # outer + inner margin
    image_height = height + margin * (2 + 2 + 3)  # outer + inner margin + banner
    # clip font size to [32, margin]
    font_size = int(margin * 200 / points_num)
    font_size = int(min(max(font_size, min_font_size), margin))
    legend_size = int(margin / 5)
    return margin, image_width, image_height, font_size, legend_size


def draw_legend(draw, margin, image_width, image_height, legend_size, model_name, embedding_layer, color_type):
    # draw pic title
    font_size = int(margin)
    draw.text((margin, image_height - (3 * margin)),
              f'[ {model_name} ]',
              fill='#000000',
              font=get_font(legend_size * 2))

    # draw embedding layer
    draw.text((margin, image_height - int(2.3 * margin)),
              f"[ {embedding_layer} layer ]",
              fill='#000000',
              font=get_font(int(legend_size * 1.5)))

    # draw legend
    font_size = int(margin / 3)
    box_width = int(margin / 2)
    column = 4
    column_width = int(box_width * 6)

    color_map = lanuage_color_map
    if color_type == 'bucc':
        color_map = bucc_color_map

    column_size = math.ceil(len(color_map) / column)
    # font_label = get_chinese_font(font_size)
    # font_count = get_chinese_font(int(font_size / 1.7))
    for i, languages in enumerate(color_map.keys()):
        row = int(i % column_size)
        col = int(i / column_size)
        x = image_width - (column * column_width) + (col * column_width)
        y = image_height - ((column_size - row) * (box_width * 1.4)) - (margin)
        color = color_map[languages]
        # print(i, row, col, x, y, word_type, color)

        # draw box
        draw.rectangle((x, y, x + box_width, y + box_width), fill=color)
        # print(word_type, color)

        # draw label
        draw.text((x + box_width * 1.5, int(y - font_size * 0.3)),
                  f'{languages}',
                  fill='#000000',
                  font=get_font(legend_size))
        # draw count
        draw.text((x + box_width * 1.5, int(y + font_size * 1)),
                  f'({color_map[languages]})',
                  fill='#000000',
                  font=get_font(int(legend_size / 1.7)))


# model: {model_name, model, tokenizer, vocab, embeddings, embeddings_2d}
# def draw_vocab_embeddings(model, charsets, width=8000, height=8000, is_detail=False, debug=False):
def draw_embeddings(model_name: str, embeddings_2d: List[List[float]], language:list, counts:list, compression_method:str,
                    embedding_layer:int, width=8000, height=8000, is_detailed=False, debug=False):
    points_num = len(embeddings_2d)

    # validation
    if sum(counts) != points_num:
        print(counts, language, "total points num:", len(embeddings_2d))
        return

    # calculate image size, margin, etc.
    margin, image_width, image_height, font_size, legend_size = calculate_size(width, height, points_num, min_font_size=68)

    # normalize embeddings
    scaler = MinMaxScaler()
    embeddings_2d_norm = scaler.fit_transform(embeddings_2d)
    if debug:
        print(embeddings_2d_norm.shape)

    # draw embeddings
    image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, width + (3 * margin), height + (3 * margin)), fill='#F0F0F0')

    # CharsetClassifier
    # classifier = CharsetClassifier(charsets=charsets, is_detailed=is_detailed)
    # word_type_count = {k: 0 for k in classifier.get_types()}
    # palette = classifier.get_palette(with_prefix_palette=True)

    if debug:
        # print(f"palette: {palette}")
        print(f"[{model_name}]: draw embedding point: {points_num}")

    # draw embedding point
    if debug:
        print("font_size:", font_size, " margin size:", margin, " legend size:", legend_size)
    # zh_font = get_chinese_font(font_size)
    if debug:
        print(f"font size: {font_size}, font: 暂时不用特殊中文")

    prev = 0
    k = 0
    for i, [x, y] in enumerate(embeddings_2d_norm):
        # 更新所属的颜色字段
        if counts[k] + prev <= i:
            prev += counts[k]
            k += 1

        word = str(int(i-prev))
        word_type = 'en'
        # word_type_count[word_type] += 1

        # if word.startswith('##'):
        #     word_type = '##' + word_type
        color = lanuage_color_map[language[k]]

        # draw text
        x = x * width + margin * 2
        y = y * height + margin * 2
        try:
            draw.text((x, y), word, fill=color, stroke_width=1, stroke_fill='#F0F0F0', font=get_font(font_size))
        except Exception as e:
            print(f"[{model_name}]: warning: draw text error: {e}")

    if debug:
        print(f"[{model_name}]: token type: {language}")

    # draw model name
    font_size = int(margin)
    draw.text((margin, image_height - (3 * margin)),
              f'[ {model_name} ]',
              fill='#000000',
              font=get_font(legend_size*2))

    # draw embedding layer
    draw.text((margin, image_height - int(2.3 * margin)),
              f"[ {embedding_layer} layer ]",
              fill='#000000',
              font=get_font(int(legend_size*1.5)))

    # # draw vocab size
    # draw.text((margin, image_height - int(1.8 * margin)),
    #           "[ vocab size: {:,} ]".format(vocab_size),
    #           fill='#000000',
    #           font=get_english_font(int(font_size / 1.5)))

    # draw legend
    font_size = int(margin / 3)
    box_width = int(margin / 2)
    column = 4
    column_width = int(box_width * 6)
    column_size = math.ceil(len(lanuage_color_map) / column)
    # font_label = get_chinese_font(font_size)
    # font_count = get_chinese_font(int(font_size / 1.7))
    for i, language in enumerate(lanuage_color_map.keys()):
        row = int(i % column_size)
        col = int(i / column_size)
        x = image_width - ((column) * column_width) + (col * column_width)
        y = image_height - ((column_size - row) * (box_width * 1.4)) - (margin)
        color = lanuage_color_map[language]
        # print(i, row, col, x, y, word_type, color)

        # draw box
        draw.rectangle((x, y, x + box_width, y + box_width), fill=color)
        # print(word_type, color)

        # draw label
        draw.text((x + box_width * 1.5, int(y - font_size * 0.3)),
                  f'{language}',
                  fill='#000000',
                  font=get_font(legend_size))
        # draw count
        draw.text((x + box_width * 1.5, int(y + font_size * 1)),
                  f'({lanuage_color_map[language]})',
                  fill='#000000',
                  font=get_font(int(legend_size/1.7)))

    return image




def draw_tatoeba_pic(model_name: str, lang_embeddings_2d: List[List[float]], languages: list, compression_method:str,
                     embedding_layer:int, width=8000, height=8000, is_detailed=False, debug=False):
    '''
    :param model_name:
    :param lang_embeddings_2d: list of 2d array
    :param languages: list of str represent language
    :param counts: languages * 2S
    :param compression_method:
    :param embedding_layer: int
    :param width:
    :param height:
    :param is_detailed:
    :param debug:
    :return: an image, text i represents the ith textfile it's extracted from, color c represents its language
    '''
    points_num = len(lang_embeddings_2d)
    if points_num != 1000*2*len(languages):
        print('dimension error!')
        return
    if debug:
        print(f"[{model_name}]: draw embedding point: {points_num}")

    # calculate image size, margin, etc.
    margin, image_width, image_height, font_size, legend_size = calculate_size(width, height, points_num)
    # draw embedding point
    if debug:
        print("font_size:", font_size, " margin size:", margin, " legend size:", legend_size)

    # normalize embeddings
    scaler = MinMaxScaler()
    embeddings_2d_norm = scaler.fit_transform(lang_embeddings_2d)
    if debug:
        print(embeddings_2d_norm.shape)

    # draw embeddings
    image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, width + (3 * margin), height + (3 * margin)), fill='#F0F0F0')

    # text is i//2000
    # language is eng (i//1000)%2 == 1 else languages[i//2000]
    for i, [x, y] in enumerate(embeddings_2d_norm):
        # 更新所属的颜色字段

        word = str(i//2000)
        lang = 'en' if (i % 2000) >= 1000 else lang3_dict[languages[i//2000]]
        color = lanuage_color_map[lang]

        # draw text
        x = x * width + margin * 2
        y = y * height + margin * 2
        try:
            draw.text((x, y), word, fill=color, stroke_width=1, stroke_fill='#F0F0F0', font=get_font(font_size))
        except Exception as e:
            print(f"[{model_name}]: warning: draw text error: {e}")

    if debug:
        print(f"[{model_name}]: token type: {languages}")

    # draw legend
    draw_legend(draw, margin=margin, image_width=image_width, image_height=image_height, legend_size=legend_size,
                model_name=model_name, embedding_layer=embedding_layer)
    return image


def draw_bucc_pic(model_name: str, lang_embeddings_2d: List[List[float]], languages: list, counts: list, compression_method:str,
                     embedding_layer:int, width=2000, height=2000, is_detailed=False, debug=False):
    '''
    :param model_name:
    :param lang_embeddings_2d: list of 2d array
    :param languages: list of str represent language
    :param counts: [x1, x1, x2, x3] counts of sents of a language
    :param compression_method:
    :param embedding_layer: int
    :param width:
    :param height:
    :param is_detailed:
    :param debug:
    :return: an image, text i represents the ith textfile it's extracted from, color c represents its language
    '''
    points_num = len(lang_embeddings_2d)
    if points_num != sum(counts):
        print(f'dimension error! embedding length:{points_num}, counts"{counts}')
        return
    if debug:
        print(f"[{model_name}]: draw embedding point: {points_num}")

    # calculate image size, margin, etc.
    margin, image_width, image_height, font_size, legend_size = calculate_size(width, height, points_num)
    # draw embedding point
    if debug:
        print("font_size:", font_size, " margin size:", margin, " legend size:", legend_size)

    # normalize embeddings
    scaler = MinMaxScaler()
    embeddings_2d_norm = scaler.fit_transform(lang_embeddings_2d)

    # draw embeddings
    image = Image.new('RGB', (image_width, image_height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, width + (3 * margin), height + (3 * margin)), fill='#F0F0F0')

    # draw point
    # language is eng (i//1000)%2 == 1 else languages[i//2000]
    # j means language index
    j = 0
    index = counts[j]
    for i, [x, y] in enumerate(embeddings_2d_norm):
        # 更新所属的颜色字段
        index -= 1
        if index < 0:
            j += 1
            index = counts[j]
        # print(index, j)
        lang = languages[j]
        color = bucc_color_map[lang]

        # draw text
        x = x * width + margin * 2
        y = y * height + margin * 2
        try:
            draw.point((x, y), fill=color)
            # draw.text((x, y), word, fill=color, stroke_width=1, stroke_fill='#F0F0F0', font=get_font(font_size))
        except Exception as e:
            print(f"[{model_name}]: warning: draw text error: {e}")

    if debug:
        print(f"[{model_name}]: token type: {languages}")

    # draw legend
    draw_legend(draw, margin=margin, image_width=image_width, image_height=image_height, legend_size=legend_size,
                model_name=model_name, embedding_layer=embedding_layer, color_type='bucc')
    return image
