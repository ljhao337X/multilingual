from collections import defaultdict
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score

def eval_lareqa(q_dict, cand_dict, languages):
    """
    Compute the averaged mAP across all languages.
    """
    score = 0
    cnt = 0
    result = {}
    for lan in tqdm(languages):
        cnt_lan = 0
        score_lan = 0
        dots = np.matmul(q_dict[lan]["embeds"], np.transpose(cand_dict["embeds"]))
        n_q = len(q_dict[lan]["qid"])
        for i in range(n_q):
            qid = q_dict[lan]["qid"][i]
            labels = [1 if qid in id_set else 0 for id_set in cand_dict["qid"]]
            tmp = average_precision_score(labels, dots[i, :])
            score += tmp
            score_lan += tmp
            cnt += 1
            cnt_lan += 1
        result[lan] = score_lan/cnt_lan
    result["all"] = score/cnt
    return result


def main():
    # key: language name, val: language information of size d*20, where d is the representation size
    lan_info = {}

    # batch size
    b_size = 128
    for k, v in tqdm(text_wiki.items()):
        v = np.asarray(v)
        emb = []
        for i in range(0, len(v), b_size):
            emb.append(question_encoder(input=tf.constant(v[i:i + b_size]))["outputs"].numpy())
        emb = np.concatenate(emb)
        u, _, _ = np.linalg.svd(np.transpose(emb), full_matrices=False)
        lan_info[k] = u[:, :20]

    lan_info["zh"] = lan_info.pop("zh-cn")


    lan_list = ["ar", "de", "el", "en", "es", "hi", "ru", "th", "tr", "vi", "zh"]

    # For mlqa-r
    # lan_list = ["ar", "de", "en", "es", "hi", "vi", "zh"]

    questions = {}
    for lan in lan_list:
        questions[lan] = defaultdict(list)

    # key: qid, embeds
    candidates = defaultdict(list)

    # memorize the corresponding index of each language in "candidates"
    index = {}

    # load dataset
    dataset_name = "xquad-r"
    # dataset_name = "mlqa-r"

    start_ = 0
    candidates["sentences"] = []
    candidates["context"] = []
    candidates["qid"] = []

    for lan in lan_list:
        questions[lan]["qid"] = []
        questions[lan]["question"] = []
        f = open(f"./dataset/{dataset_name}/{lan}.json", encoding='utf-8')
        data = json.load(f)["data"]
        for entry in data:
            for para in entry["paragraphs"]:
                n_sents = len(para["sentences"])
                context = [para["context"]] * n_sents
                qid_list = [qs["id"] for qs in para["qas"]]
                q_list = [qs["question"] for qs in para["qas"]]
                sent_qid_list = [set() for _ in range(n_sents)]
                for qs in para["qas"]:
                    a_start = qs["answers"][0]["answer_start"]
                    for i in range(n_sents):
                        if a_start >= para["sentence_breaks"][i][0] and a_start <= para["sentence_breaks"][i][1]:
                            sent_qid_list[i].add(qs["id"])
                            break
                candidates["sentences"].extend(para["sentences"])
                candidates["context"].extend(context)
                candidates["qid"].extend(sent_qid_list)
                questions[lan]["qid"].extend(qid_list)
                questions[lan]["question"].extend(q_list)
        index[lan] = [start_, len(candidates["sentences"])]
        start_ = len(candidates["sentences"])
        f.close()

    # encode dataset
    for lan in lan_list:
        for k, v in questions[lan].items():
            if k != "qid":
                questions[lan][k] = np.asarray(v)

    for k, v in candidates.items():
        if k != "qid":
            candidates[k] = np.asarray(v)

    for lan in tqdm(lan_list):
        q_lan = questions[lan]["question"]
        for i in range(0, len(q_lan), b_size):
            embeds_ = question_encoder(
                input=tf.constant(q_lan[i:i + b_size]))["outputs"]
            questions[lan]["embeds"].append(embeds_.numpy())

    for i in tqdm(range(0, len(candidates["sentences"]), b_size)):
        embeds_ = response_encoder(
            input=tf.constant(candidates["sentences"][i:i + b_size]),
            context=tf.constant(candidates["context"][i:i + b_size]))["outputs"]
        candidates["embeds"].append(embeds_.numpy())

    for lan in lan_list:
        questions[lan]["embeds"] = np.concatenate(questions[lan]["embeds"])
    candidates["embeds"] = np.concatenate(candidates["embeds"])

    result_no_lir = eval_lareqa(questions, candidates, lan_list)

    map_no_lir = result_no_lir["all"]
    print(f"Before applying LIR, averaged mAP is: {map_no_lir}")


if __name__ == '__main__':
    main()

