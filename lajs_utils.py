''' 
_*_ coding: utf-8 _*_
Date: 2021/8/14
Author: 
Intent:
'''

import json, os, re, math
from collections import defaultdict
import pandas as pd
import jieba
import numpy as np

try:
    from gensim.summarization import bm25
except:
    pass


# SMALL_DEV_QUERY_ID = ['221','259','1325','1355','1405','1430','1972','1978','2132','2143',
#                       '2331','2361','2373','3228','3342','3746','3765','4738','4794','4829']

# def get_ajName(query_file, candidate_dir,label_top30_file, merge_excel):
#     excel_data = defaultdict(list)
#
#     with open(label_top30_file, 'r', encoding='utf8') as f:
#         all_label_top30 = json.load(f)
#
#     with open(query_file, 'r', encoding='utf8') as f:
#         for line in f:
#             query = json.loads(line.strip())
#             qidx, q, crime = str(query['ridx']), query['q'], query['crime']
#             query_str = 'id: ' + qidx + '\n' + 'query: ' + q + '\n' + 'crime: ' + str(crime)
#
#             single_label_top30 = all_label_top30[qidx].items()
#             single_label_top30 = sorted(single_label_top30, key=lambda x:x[1], reverse=True)
#             names = []
#             for didx, relevence in single_label_top30:
#                 doc_path = os.path.join(candidate_dir, qidx, didx+'.json')
#                 with open(doc_path, 'r', encoding='utf8') as fd:
#                     doc = json.load(fd)
#                 names.append(didx + ' **** ' + doc['ajName'] + ' **** ' + str(relevence))
#             name_str = '\n'.join(names)
#
#             excel_data['query'].append(query_str)
#             excel_data['ajName_rel'].append(name_str)
#
#     df = pd.DataFrame.from_dict(excel_data)
#     df.to_excel(merge_excel)


def cut_doc(doc):
    return [s for s in re.split(r'[。！；？]', doc) if len(s) > 0]


def get_stopwords(stw_file):
    with open(stw_file, 'r', encoding='utf8') as g:
        words = g.readlines()
    stopwords = [i.strip() for i in words]
    stopwords.extend(['.','（','）','-','×'])
    return stopwords


def search_for_related_sents(query, sents_list, stopwords=None, select=1):
    corpus = []
    stopwords = [] if stopwords is None else stopwords
    for sent in sents_list:
        sent_tokens = [w for w in jieba.lcut(sent) if w not in stopwords]
        corpus.append(sent_tokens)
    bm25model = bm25.BM25(corpus)

    q_tokens = [w for w in jieba.lcut(query) if w not in stopwords]
    scores = bm25model.get_scores(q_tokens)
    rank_index = np.array(scores).argsort().tolist()[::-1]
    rank_index = rank_index[:select]
    return [sents_list[i] for i in rank_index], [scores[i] for i in rank_index]


def select_relavence(query_file, candidate_dir, stw_file, saved_file, sent_group=5, select=1):
    queries = []
    stopwords = get_stopwords(stw_file)
    with open(query_file, 'r', encoding='utf8') as f:
        for line in f:
            queries.append(json.loads(line.strip()))

    data = []
    for query in queries:
        qidx, q, crime = str(query['ridx']), str(query['q']), '、'.join(query['crime'])

        # if qidx in SMALL_DEV_QUERY_ID:
        #     continue

        doc_dir = os.path.join(candidate_dir, qidx)
        doc_files = os.listdir(doc_dir)
        selected_docs = []
        for doc_file in doc_files:
            doc_path = os.path.join(doc_dir, doc_file)
            didx = str(doc_file.split('.')[0])
            with open(doc_path, 'r', encoding='utf8') as fd:
                sample_d = json.load(fd)
            doc, d_crime = sample_d['ajjbqk'], sample_d['ajName']
            sents_list = cut_doc(doc)
            pre_sents = sents_list[:sent_group*2]

            # # 用query中的每个句子去搜索更有效果
            # q_sents_list = cut_doc(queries[1]['q'])
            # for q_s in q_sents_list:
            #     rel_sents, scores = search_for_related_sents(q_s, sents_list, stopwords, select=select)
            #     print('qqq: ', q_s)
            #     print('sss_qqq', rel_sents)
            #     print('score: ', scores)

            group_sents = []
            for i in range(0, len(sents_list), sent_group // 2):
                if i == 0 or len(sents_list[i:i+sent_group]) >= sent_group:
                    group_sents.append('。'.join(sents_list[i:i+sent_group]))
                if i + sent_group > len(sents_list):
                    break
            print(len(group_sents))
            rel_sents, scores = search_for_related_sents(q, group_sents, stopwords, select=select)
            for s in cut_doc(rel_sents[0]):
                if s not in pre_sents:
                    pre_sents.append(s)
            pre_sents = '。'.join(pre_sents)
            # print(rel_sents)
            # print(scores)
            # print(pre_sents)
            selected_docs.append({'didx':didx, 'd_crime':d_crime, 'content':pre_sents})
        data.append({'qidx':qidx, 'q':q, 'crime':crime, 'docs':selected_docs})

    with open(saved_file, 'w', encoding='utf8') as fs:
        json.dump(data, fs, ensure_ascii=False, indent=2)


# def add_label_to_file(data_file, label_top30_file, new_data_file):
#     with open(label_top30_file, 'r', encoding='utf8') as f:
#         all_label_top30 = json.load(f)
#
#     with open(data_file, 'r', encoding='utf8') as f1:
#         data = json.load(f1)
#
#     new_data = []
#     for s in data:
#         qidx = s['qidx']
#         s['labels'] = all_label_top30[qidx]
#         new_data.append(s)
#
#     with open(new_data_file, 'w', encoding='utf8') as f2:
#         json.dump(new_data, f2, ensure_ascii=False, indent=2)


def ndcg(ranks, K):
    dcg_value = 0.
    idcg_value = 0.

    sranks = sorted(ranks, reverse=True)

    for i in range(0,K):
        logi = math.log(i+2,2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi
    if idcg_value == 0.0:
        idcg_value += 0.00000001
    return dcg_value/idcg_value


# def count_tokens():
#     file = './data/large_train.json'
#     with open(file, 'r', encoding='utf8') as f:
#         data = json.load(f)
#     max_wn = 0
#     all_wn = 0
#     count = 0
#     for s in data:
#         q = s['q'] + s['crime']
#         docs = s['docs']
#         for d in docs:
#             doc = d['d_crime'] + d['content']
#             wn = len(q) + len(doc)
#             if wn > 2560:
#                 print(1)
#             if wn > max_wn:
#                 max_wn = wn
#             all_wn += wn
#             count += 1
#     print(f'最长字数：{max_wn}, 平均字数：{all_wn/count}')
#
#
# def get_model_result(dev_file, predict_file, label_file, result_file):
#     with open(dev_file, 'r', encoding='utf8') as f:
#         dev_data = json.load(f)
#     dev_dict = {}
#     for s in dev_data:
#         dev_dict[s['qidx']] = s
#     with open(predict_file, 'r', encoding='utf8') as f:
#         preds = json.load(f)
#     with open(label_file, 'r', encoding='utf8') as f:
#         labels = json.load(f)
#     assert len(preds) == len(dev_dict)
#     for qidx, cands in preds.items():
#         cands = cands[:30]
#         dev_docs = {d['didx']: d for d in dev_dict[qidx]['docs']}
#         d_labels = labels[qidx]
#
#         new_dev_docs = []
#         for didx in cands:
#             didx = str(didx)
#             temp_d = dev_docs[didx]
#             temp_d['relevance'] = d_labels[didx] if didx in d_labels else 0
#             new_dev_docs.append(temp_d)
#         dev_dict[qidx]['docs'] = new_dev_docs
#
#     with open(result_file, 'w', encoding='utf8') as f:
#         json.dump(dev_dict, f, ensure_ascii=False, indent=2)

#
# def merge_data_file(train_file, dev_file, new_train_file):
#     with open(train_file, 'r', encoding='utf8') as ft:
#         train_data = json.load(ft)
#     with open(dev_file, 'r', encoding='utf8') as fd:
#         dev_data = json.load(fd)
#     merge_data = train_data + dev_data
#     with open(new_train_file, 'w', encoding='utf8') as fm:
#         json.dump(merge_data, fm, ensure_ascii=False, indent=2)


# def ensemble(rbt3_pred_file, bert_only256_pred_file, rbt3_ratio=0.8):
#     with open(rbt3_pred_file, 'r', encoding='utf8') as f:
#         rbt3_p = json.load(f)
#     with open(bert_only256_pred_file, 'r', encoding='utf8') as f:
#         bert_only256_p = json.load(f)
#
#     new_res = {}
#     for qidx, rbt3_res in rbt3_p.items():
#         rbt3_ids = [x[0] for x in rbt3_res]
#         rbt3_scores = [x[1] for x in rbt3_res]
#         nor_rbt3_scores = normalize(rbt3_scores)
#         nor_rbt3_res = {rbt3_ids[i]: nor_rbt3_scores[i] for i in range(len(rbt3_ids))}
#         # print(rbt3_scores)
#         # print(nor_rbt3_scores)
#
#         bert_ids = [x[0] for x in bert_only256_p[qidx]]
#         bert_scores = [x[1] for x in bert_only256_p[qidx]]
#         nor_bert_scores = normalize(bert_scores)
#         # print(bert_scores)
#         # print(nor_bert_scores)
#
#         ens_scores = [nor_bert_scores[j] * (1-rbt3_ratio) + nor_rbt3_res[bert_ids[j]] * rbt3_ratio
#                       for j in range(len(nor_bert_scores))]
#         ens_res = list(zip(bert_ids, ens_scores))
#         ens_res.sort(key=lambda x: x[1], reverse=True)
#         new_res[qidx] = ens_res
#     return new_res


# def cal_ndcg(rbt3_ratio):
#     dev_top30_label_file = './data/small/label_top30_dict.json'
#     with open(dev_top30_label_file, 'r', encoding='utf8') as f:
#         all_labels = json.load(f)
#
#     # ensemble
#     # rbt3_pred_file = './prediction_rbt3_PI.json'
#     # bert_only256_pred_file = './prediction_bert_only256.json'
#     # all_preds = ensemble(rbt3_pred_file, bert_only256_pred_file, rbt3_ratio=rbt3_ratio)
#     # print(all_labels)
#
#     with open('./test_result/PI3_seg_0_1/prediction.json', 'r', encoding='utf8') as f:
#         all_preds = json.load(f)
#
#     ndcgs = []
#     # for qidx, pred_res in all_preds.items():
#     for qidx, pred_ids in all_preds.items():
#         # pred_ids = [x[0] for x in pred_res]
#         did2rel = all_labels[qidx]
#         ranks = [did2rel[idx] if idx in did2rel else 0 for idx in pred_ids]
#         ndcgs.append(ndcg(ranks, 30))
#         print(f'********** qidx: {qidx} **********')
#         print(f'top30 pred_ids: {pred_ids}')
#         print(f'ranks: {ranks}')
#     print(ndcgs)
#     return sum(ndcgs) / len(ndcgs)


def normalize(scores):
    scores = np.array(scores)
    miu, sigma = np.mean(scores), np.std(scores)
    if sigma == 0.0:
        sigma += 0.000001
    nor_scores = (scores - miu) / sigma
    return nor_scores


def doc_segment_selection(data_file, model_path, seg_sel_file):
    from lajs_model import FirstModel
    from lajs_config import LajsConfig
    import torch
    from lajs_datasets import SegmentSelectionDataset
    from tqdm import tqdm

    def _move_to_device(batch, device):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch

    model = FirstModel(LajsConfig)
    print(f'loading model from `{model_path}`...')
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    datasets = SegmentSelectionDataset(LajsConfig)
    model.to('cuda')
    model.eval()

    with open(data_file, 'r', encoding='utf8') as fr:
        data = json.load(fr)

    with torch.no_grad():
        new_data = []
        for query_json in tqdm(data):
            q, q_crime, docs = query_json['q'], query_json['crime'], query_json['docs']
            new_docs = []
            for d in docs:
                batch = datasets.get_predict_samples(q, q_crime, d)
                batch = _move_to_device(batch, 'cuda')
                num_q, num_d = batch['num_q'], batch['num_d']
                print(num_q, num_d)
                scores = model(batch)
                scores = scores[1:].reshape(num_q, num_d)
                max_score_indexes = torch.argsort(scores, dim=-1, descending=True)
                print(scores)
                print(max_score_indexes)
                d['scores'] = scores.cpu().tolist()
                d['max_indexes'] = max_score_indexes.cpu().tolist()
                new_docs.append(d)
            query_json['docs'] = new_docs
            new_data.append(query_json)
    with open(seg_sel_file, 'w', encoding='utf8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # query_file = './data/small/query.json'
    # candidate_dir = './data/small/candidates/'
    # label_top30_file = './data/small/label_top30_dict.json'
    # merge_excel = './data/query2ajName.xlsx'
    # get_ajName(query_file, candidate_dir, label_top30_file, merge_excel)

    query_file = './data/small/query.json' #'./data/LeCaRD-main/data/query/query.json' #'./data/small/query.json'
    doc_file = './data/small/candidates/' #'./data/LeCaRD-main/data/candidates/' #'./data/small/candidates/'
    stw_file = './baseline/stopword.txt' #'./baseline/stopword.txt'
    saved_file = './data/small_dev.json' #'./data/large_train.json' #'./data/small_dev.json'
    # select_relavence(query_file, doc_file, stw_file, saved_file, sent_group=6, select=1)

    # data_file = './data/small_dev.json'
    # label_top30_file = './data/LeCaRD-main/data/label/label_top30_dict.json'
    # new_data_file = './data/small_dev-1.json'
    # # add_label_to_file(data_file, label_top30_file, new_data_file)

    # count_tokens()

    # dev_file = './data/small_dev.json'
    # predict_file = './prediction_orig.json'
    # label_file = './data/small/label_top30_dict.json'
    # result_file = './test_result/res_orig.json'
    # get_model_result(dev_file, predict_file, label_file, result_file)

    # train_file = './data/large_train_87.json'
    # dev_file = './data/small_dev.json'
    # new_train_file = './data/large_train.json'
    # merge_data_file(train_file, dev_file, new_train_file)

    data_file = './data/large_train.json'
    model_path = './model_rbt3_seg/pytorch_model.bin'
    seg_sel_file = './data/large_train_segment_selection.json'
    doc_segment_selection(data_file, model_path, seg_sel_file)

    pass