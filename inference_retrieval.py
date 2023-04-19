import os


import json
import argparse
import collections
import faiss
import langid
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from modelscope.utils.constant import ModeKeys
from modelscope.utils.logger import get_logger
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer

logger = get_logger()


def collate(batch):
    query = [item['query'] for item in batch]
    positive = [item['positive'] for item in batch]
    negative = [item['negative'] for item in batch]
    return query, positive, negative


def measure_result(result_dict):
    recall_k = [1, 5, 10, 20]
    meters = {f'R@{k}': [] for k in recall_k}

    for output, target in zip(result_dict['outputs'], result_dict['targets']):
        for k in recall_k:
            if target in output[:k]:
                meters[f'R@{k}'].append(1)
            else:
                meters[f'R@{k}'].append(0)
    for k, v in meters.items():
        meters[k] = sum(v) / len(v)
    return meters


class MyDocumentGroundedDialogRetrievalTrainer(DocumentGroundedDialogRetrievalTrainer):

    def evaluate(self, per_gpu_batch_size=32, lang='fr', checkpoint_path=None):
        """
        Evaluate testsets
        """

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path)
            self.model.model.load_state_dict(state_dict)

        valid_loader = DataLoader(
            dataset=self.eval_dataset,
            batch_size=per_gpu_batch_size,
            collate_fn=collate)
        self.model.model.eval()
        with torch.no_grad():
            all_ctx_vector = []
            for mini_batch in tqdm.tqdm(
                    range(0, len(self.all_passages), per_gpu_batch_size)):
                context = self.all_passages[mini_batch:mini_batch
                                            + per_gpu_batch_size]
                processed = \
                    self.preprocessor({'context': context},
                                        invoke_mode=ModeKeys.INFERENCE,
                                        input_type='context')
                sub_ctx_vector = self.model.encode_context(
                    processed).detach().cpu().numpy()
                all_ctx_vector.append(sub_ctx_vector)

            all_ctx_vector = np.concatenate(all_ctx_vector, axis=0)
            all_ctx_vector = np.array(all_ctx_vector).astype('float32')
            faiss_index = faiss.IndexFlatIP(all_ctx_vector.shape[-1])
            faiss_index.add(all_ctx_vector)

            results = {'outputs': [], 'targets': []}
            for index, payload in enumerate(tqdm.tqdm(valid_loader)):
                query, positive, negative = payload
                processed = self.preprocessor({'query': query},
                                                invoke_mode=ModeKeys.INFERENCE)
                query_vector = self.model.encode_query(
                    processed).detach().cpu().numpy().astype('float32')
                D, Index = faiss_index.search(query_vector, 20)
                results['outputs'] += [[
                    self.all_passages[x] for x in retrieved_ids
                ] for retrieved_ids in Index.tolist()]
                results['targets'] += positive
            meters = measure_result(results)
            result_path = os.path.join(self.model.model_dir,
                                        f'evaluate_result_{lang}.json')
            with open(result_path, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        logger.info(meters)
        return meters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_path', default='./model/retrieval/nlp_convai_retrieval_pretrain_infoxlm_baseline_v1', type=str, required=False,
                        help="cache directory") 
    parser.add_argument('--total_epoches', default=60, type=int, required=False, help="number of epochs")
    parser.add_argument('--batch_size', default=128, type=int, required=False, help="batch size")
    parser.add_argument('--per_gpu_batch_size', default=256, type=int, required=False, help="batch size for evaluate")
    parser.add_argument('--accumulation_steps', default=1, type=int, required=False, help="accumulation steps")
    parser.add_argument('--lr', default=2e-5, type=float, required=False, help="learning rate")
    # parser.add_argument('--do_drop', default=False, type=bool, required=False, help="whether to use R-drop")

    args = parser.parse_args()

    with open('test.json') as f_in:
        with open('input.jsonl', 'w') as f_out:
            for line in f_in.readlines():
                sample = json.loads(line)
                sample['positive'] = ''
                sample['negative'] = ''
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

    langid.set_languages(['fr','vi'])
    eval_dataset_fr = []
    eval_dataset_vi = []
    index_mask = []
    with open('input.jsonl') as f:
        for line in f.readlines():
            index_mask.append(langid.classify(json.loads(line)['query'])[0])
            if langid.classify(json.loads(line)['query'])[0] == 'fr':
                eval_dataset_fr.append(json.loads(line))
            if langid.classify(json.loads(line)['query'])[0] == 'vi':
                eval_dataset_vi.append(json.loads(line))

    # 分别加载两个文档集
    all_passages_fr = []
    all_passages_vi = []
    with open(f'all_passages/fr.json') as f:
        all_passages_fr += json.load(f)
    with open(f'all_passages/vi.json') as f:
        all_passages_vi += json.load(f)

    # 定义并加载两个模型
    cache_path = args.cache_path
    trainer_fr = MyDocumentGroundedDialogRetrievalTrainer(
        model=cache_path,
        train_dataset=None,
        eval_dataset=eval_dataset_fr,
        all_passages=all_passages_fr
    )
    trainer_vi = MyDocumentGroundedDialogRetrievalTrainer(
        model=cache_path,
        train_dataset=None,
        eval_dataset=eval_dataset_vi,
        all_passages=all_passages_vi
    )
    # 分别检索两种语言的相关文档
    trainer_fr.evaluate(per_gpu_batch_size=256, lang='fr',
        checkpoint_path=os.path.join(trainer_fr.model.model_dir,
                                    'finetuned_model.bin'))

    trainer_vi.evaluate(per_gpu_batch_size=256, lang='vi',
        checkpoint_path=os.path.join(trainer_vi.model.model_dir,
                                    'finetuned_model.bin'))

    # 合并两种语言的结果
    with open(f'{cache_path}/evaluate_result_fr.json') as f:
        evaluate_result_fr = json.load(f)
    with open(f'{cache_path}/evaluate_result_vi.json') as f:
        evaluate_result_vi = json.load(f)

    evaluate_result = collections.OrderedDict()
    evaluate_result['outputs'] = []
    evaluate_result['targets'] = []
    idx_fr, idx_vi, idx = 0, 0, 0
    while idx_fr < len(evaluate_result_fr['outputs']) and \
        idx_vi < len(evaluate_result_vi['outputs']):
        if index_mask[idx] == 'fr':
            evaluate_result['outputs'].append(evaluate_result_fr['outputs'][idx_fr])
            evaluate_result['targets'].append(evaluate_result_fr['targets'][idx_fr])
            idx_fr += 1
        elif index_mask[idx] == 'vi':
            evaluate_result['outputs'].append(evaluate_result_vi['outputs'][idx_vi])
            evaluate_result['targets'].append(evaluate_result_vi['targets'][idx_vi])
            idx_vi += 1
        idx += 1

    while idx_fr < len(evaluate_result_fr['outputs']):
        evaluate_result['outputs'].append(evaluate_result_fr['outputs'][idx_fr])
        evaluate_result['targets'].append(evaluate_result_fr['targets'][idx_fr])
        idx_fr += 1

    while idx_vi < len(evaluate_result_vi['outputs']):
        evaluate_result['outputs'].append(evaluate_result_vi['outputs'][idx_vi])
        evaluate_result['targets'].append(evaluate_result_vi['targets'][idx_vi])
        idx_vi += 1

    result_path = os.path.join(cache_path, 'evaluate_result.json')
    with open(result_path, 'w') as f:
        json.dump(evaluate_result, f, ensure_ascii=False, indent=4)
