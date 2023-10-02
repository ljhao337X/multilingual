from draw_pic import draw_embeddings, reduce_to_2d_mds, reduce_to_2d_pca
import os
import pickle


def output_pic(model_name, embedding_layer, embeddings_2d, compression_method, debug=False):
    image = draw_embeddings(
        model_name=model_name,
        embeddings_2d=embeddings_2d,
        embedding_layer=embedding_layer,
        language=['en', 'zh'],
        counts=[len(embeddings_2d) / 2, len(embeddings_2d) / 2],
        compression_method=compression_method,
        width=8000,
        height=8000,
        debug=debug
    )

    folder = './output/test/'
    # 生成文件名
    filename = model_name.replace('/', '_') + '_' + compression_method + f'.{embedding_layer}.jpg'
    filename = 'embeddings_' + filename
    if folder is not None and len(folder) > 0:
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, filename)

    # save to file
    if debug:
        print(f"[{model_name}]: save {embedding_layer}_embeddings to {filename}...")
    image.save(filename, quality=80, optimize=True, progressive=True)


if __name__ == '__main__':
    debug = True
    # features_out_hook = np.random.random((12, 1024*10, 2))
    f = open('./tmp_ERNIE-M.pkl', 'rb')
    features_out_hook = pickle.loads(f.read())
    f.close()
    if debug:
        print(len(features_out_hook), features_out_hook[0].shape)

    modelName = 'ERNIE-M'
    # compressionMethod = 'mds'

    for i in range(0, 12):
        try:
            embeddings2d = reduce_to_2d_mds(features_out_hook[i])
            output_pic(modelName, i, embeddings2d, compression_method='mds', debug=debug)

            embeddings2d = reduce_to_2d_pca(features_out_hook[i])
            output_pic(modelName, i, embeddings2d, compression_method='pca', debug=debug)
        except:
            print(f'layer:{i} is not symmetric?')
            pass





    # embeddings2d = reduce_to_2d_mds(features_out_hook[10], debug)
    # output_pic(modelName, 10, embeddings2d, compression_method=compressionMethod, debug=debug)
