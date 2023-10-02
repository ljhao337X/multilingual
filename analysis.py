import numpy as np
import os

ACC_PATH = './output/accuracy_XLM-A/'
lang = []

def analysis_best_accuracy(file):
    # file = './output/accuracy_INFOXLM/to_eng_to_{}.npy'.format(lang1)
    accuracy = np.load(file)
    # # (26,2) -> to_eng 13, to_lang1 13; 2: accuracy_first, accuracy_any(5)
    # print(accuracy.shape)
    accuracy_to_eng = accuracy[0:13, 0].squeeze()
    accuracy_to_lang1 = accuracy[13:26, 0].squeeze()
    eng_gap = accuracy_to_eng - accuracy_to_lang1
    gap_most = np.argmax(eng_gap)
    to_eng_best, to_lang1_best = np.argmax(accuracy_to_eng), np.argmax(accuracy_to_lang1)
    if to_eng_best != to_lang1_best:
        print("dismatch in " + os.path.basename(file))
    if to_eng_best==0 or to_lang1_best == 0:
        lang.append(os.path.basename(file)[-7:-4])
        print("embedding got the best in " + os.path.basename(file))
    return np.mean(eng_gap), gap_most, to_eng_best, to_lang1_best


if __name__ == '__main__':
    eng_best = [0 for i in range(13)]
    lang_best = eng_best.copy()
    gap = []
    gap_most = eng_best.copy()
    for file in os.listdir(ACC_PATH):
        x, k, i, j = analysis_best_accuracy(ACC_PATH + file)
        gap.append(x)
        gap_most[k] += 1
        eng_best[i] += 1
        lang_best[i] += 1

    print(sum(gap)/len(gap), gap_most, eng_best, lang_best)
    print("languages where embedding 0 perform the best: ", lang)

