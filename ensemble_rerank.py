"""
作者：ZZN
日期：2023年03月16日
"""
# from modelscope.msdatasets import MsDataset
# from modelscope.utils.constant import DownloadMode

# dataset = MsDataset.load(
#     'DAMO_ConvAI/EnDoc2BotDialogue',
#     download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

import copy
import json
from collections import Counter

# 利用BM25构造检索数据集
from rank_bm25 import *


def dragpassage_trans_rerank():
    with open('./all_passages/id_to_passage.json') as f:
        id_to_passage = json.load(f)
    with open(f'./model/rerank/output_rerank_result/rerank_output_10.jsonl') as f_in:
        statistics = {json.loads(letter)['id']: [] for letter in f_in.readlines()}
    statistics_final = copy.deepcopy(statistics)
    all_train_dataset = {}
    num_type = ['14', '10', '11', '12', '13', '15',
                '16', '17', '18', '19', '21', ]
    for num in num_type:
        with open(f'./model/rerank/output_rerank_result/rerank_output_{num}.jsonl') as f_in:
            co_aug_dataset = []
            for line in f_in.readlines():
                line_temp = json.loads(line)
                if num == '14':  # 5是最好
                    statistics[line_temp['id']].append(line_temp['scored_pids'][0][0])
                    statistics[line_temp['id']].append(line_temp['scored_pids'][0][0])
                    statistics[line_temp['id']].append(line_temp['scored_pids'][0][0])
                    statistics[line_temp['id']].append(line_temp['scored_pids'][0][0])
                statistics[line_temp['id']].append(line_temp['scored_pids'][0][0])
                co_aug_dataset.append(json.loads(line))
        all_train_dataset[num] = co_aug_dataset
    for k, v in statistics.items():
        count_dict = Counter(v)
        for k_d, v_d in count_dict.items():
            statistics_final[k].append(str(k_d) + ':' + str(v_d) + ',' + id_to_passage[k_d])
    for k, v in statistics.items():
        count_dict = Counter(v)
        all_train_dataset['14'][int(k)]['scored_pids'][0][0] = count_dict.most_common(1)[0][0]
        all_train_dataset['14'][int(k)]['output'][0]['provenance'][0]['wikipedia_id'] = count_dict.most_common(1)[0][0]

    print(f"all_train_dataset{len(all_train_dataset)}")

    # with open('/model/rerank//rerank_statistics.json', 'w') as f_out:
    #     f_out.write(json.dumps(statistics_final, ensure_ascii=False, indent=4))
    with open('./model/rerank/rerank_output.jsonl', 'w') as f_out:
        for every_dict in all_train_dataset['14']:
            f_out.write(json.dumps(every_dict) + '\n')


if __name__ == '__main__':
    dragpassage_trans_rerank()
