import os
import json
import argparse
from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_names',
                        default=["mBart-Large-transpretrain-4rerank/Vi_Fr_pretrain_on_Vi_Fr_En2Vi_En2Fr_finetuned_model.bin",
                                #  'mBart-Large-transpretrain-6-4rerank/Vi_Fr_pretrain_on_En_Zh_En2Vi_En2Fr_Cn2Vi_Cn2Fr_finetuned_model.bin',
                                # "pretrain-325-4rerank/Vi_Fr_pretrain_on_En_Zh_En2Vi_En2Fr_Cn2Vi_Cn2Fr_finetuned_model.bin",
                                 ]
                        , type=str, nargs='+', required=False,
                        help='model name to use')  # Vi_Fr_pretrain_on_Vi_Fr_En2Vi_En2Fr_finetuned_model.bin
    parser.add_argument('--model_res_num', default=[5, ], type=int, nargs='+', required=False,
                        help='each model beam search num')
    parser.add_argument('--file_index', default=[1,], type=int, nargs='+', required=False, )
    parser.add_argument('--output_dir', default='final_rerank_data', type=str,
                        required=False,
                        help="output_dir model directory")
    parser.add_argument('--cache_dir', default='/root/autodl-tmp/generation', type=str, required=False,
                        help="cache directory")  # '../DAMO-ConvAI/acl23doc2dial'  /root/autodl-tmp
    args = parser.parse_args()

    dataset = []
    index = []
    # '../pretrain_data/corespond_data.jsonl'
    with open('../all_passages/id_to_passage.json') as f:
        id_to_passage = json.load(f)
    # with open('../pretrain_data/corespond_data.jsonl') as f:
    #     for line in f.readlines():
    #         data = json.loads(line)
    #         if len(data['rerank']) < 5:
    #             continue
    #         dataset.append(data)
    with open('../rerank_output.jsonl') as f:
        for line in f.readlines():
            data = json.loads(line)
            for j in range(5):
                dataset.append({
                    'query': data['input'],
                    'rerank': json.dumps([id_to_passage[data['output'][0]['provenance'][j]['wikipedia_id']]], ensure_ascii=False),
                    'response': '<response> @'
                })
    # dataset = dataset[:102]

    model_nums = len(args.model_names)
    cache_path = os.path.join(args.cache_dir, args.output_dir)
    model_res_num = args.model_res_num

    model_names = args.model_names
    predictions = []

    prediction = [dataset[j]['response'] for j in range(len(dataset))]
    # predictions.append(prediction) # golden


    for idx, model_name in enumerate(model_names):
        result_name = f'evaluate_result_{str(args.file_index[idx])}.json'

        with open(f'{cache_path}/{result_name}') as f:
            prediction = json.load(f)['outputs']

        prediction = [[prediction[j] for j in range(i, len(prediction), model_res_num[idx])] for i in
                      range(model_res_num[idx])]
        for i in range(model_res_num[idx]):
            predictions.append(prediction[i])

    hard_negative = []
    idx = 0
    for i in range(len(predictions[0])):
        data = []
        for j in range(len(predictions)):
            query = dataset[idx]['query']
            response = predictions[j][i]

            passage = json.loads(dataset[idx]['rerank'])[0]
            idx += 1
            data.append({
                'response': response.replace('<response>', '').strip(),
                'query': query,
                'passage': passage
            })
        hard_negative.append(data)
    print("")
    # for i in range(len(predictions[0])):
    #     data = []
    #     for j in range(len(predictions)):
    #         query = dataset[i]['query']
    #         response = predictions[j][i]
    #         if j==0:
    #             passage = id_to_passage[dataset[i]['rerank'][0]['wikipedia_id']]
    #         else:
    #             passage = id_to_passage[dataset[i]['rerank'][j]['wikipedia_id']]
    #         data.append({
    #             'response': response.replace('<response>', '').strip(),
    #             'query': query,
    #             'passage': passage
    #         })
    #     hard_negative.append(data)
    # print("")
    with open('../pretrain_data/to_rerank.json', 'w') as f:
        json.dump(hard_negative, f, ensure_ascii=False)
