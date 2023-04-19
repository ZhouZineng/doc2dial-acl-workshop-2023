import argparse
import json
import os
import random
from typing import Any, Dict, Union
import csv

import torch
import torch.cuda
import torch.nn.functional as F
from modelscope.models.nlp import DocumentGroundedDialogRerankModel
from modelscope.msdatasets import MsDataset
from modelscope.pipelines.nlp import DocumentGroundedDialogRerankPipeline
from modelscope.preprocessors.nlp import \
    DocumentGroundedDialogRerankPreprocessor
from modelscope.trainers.nlp.document_grounded_dialog_rerank_trainer import \
    DocumentGroundedDialogRerankTrainer
from modelscope.utils.constant import DownloadMode
from modelscope.utils.logger import get_logger
from tqdm import tqdm

logger = get_logger()


def to_distinct_doc_ids(passage_ids):
    doc_ids = []
    for pid in passage_ids:
        # MARK
        doc_id = pid
        if doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


class myDocumentGroundedDialogRerankPipeline(DocumentGroundedDialogRerankPipeline):
    def __init__(self,
                 model: Union[DocumentGroundedDialogRerankModel, str],
                 preprocessor: DocumentGroundedDialogRerankPreprocessor = None,
                 config_file: str = None,
                 device: str = 'cuda',
                 auto_collate=True,
                 seed: int = 88,
                 **kwarg):
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate,
            seed=seed,
            **kwarg
        )

    def measure_result(self, label_dict):
        recall_k = [1, 2, 3, 5]
        meters = {f'R@{k}': [] for k in recall_k}
        for every_dict, target in zip(self.guess, label_dict):
            target = target
            for k in recall_k:
                out_dict = [p['wikipedia_id'] for p in every_dict['output'][0]['provenance'][:k]]
                if target in out_dict:
                    meters[f'R@{k}'].append(1)
                else:
                    meters[f'R@{k}'].append(0)
        for k, v in meters.items():
            meters[k] = sum(v) / len(v)
        logger.info(meters)
        return {'R@1': meters['R@1']}

    def forward(self, dataset: Union[list, Dict[str, Any]],
                **forward_params) -> Dict[str, Any]:
        self.guess = []
        self.model.eval()
        with torch.no_grad():
            for jobj in dataset:
                inst_id = jobj['id']
                probs = self.one_instance(jobj['input_ids'],
                                          jobj['attention_mask'])
                passages = jobj['passages']
                query = jobj['query']
                scored_pids = [(p['pid'], prob)
                               for p, prob in zip(passages, probs)]
                scored_pids.sort(key=lambda x: x[1], reverse=True)
                wids = to_distinct_doc_ids([
                    pid for pid, prob in scored_pids
                ])  # convert to Wikipedia document ids
                pred_record = {
                    'id':
                        inst_id,
                    'input':
                        query,
                    'scored_pids':
                        scored_pids,
                    'output': [{
                        'answer':
                            '',
                        'provenance': [{
                            'wikipedia_id': wid
                        } for wid in wids]
                    }]
                }
                if self.args['include_passages']:
                    pred_record['passages'] = passages
                self.guess.append(pred_record)
        # if args['kilt_data']:
        #     evaluate(dataset, args['output'])


class myDocumentGroundedDialogRerankTrainer(DocumentGroundedDialogRerankTrainer):
    def __init__(self, model, eval_dataset, train_dataset, **args):
        super().__init__(model, train_dataset, **args)
        self.epoch_now = -1
        self.eval_dataset = eval_dataset
        self.output_dir = args['args']['output_dir']

    def train(self):
        rand = random.Random()
        losses = []
        best_score = 0.0
        while self.optimizer.should_continue():
            self.epoch_now += 1
            self.optimizer.model.train()
            dataset = block_shuffle(self.dataset, block_size=100000, rand=rand)
            for line_ndx, jobj in enumerate(dataset):
                inst_id = jobj['id']
                if inst_id not in self.inst_id2pos_pids:
                    continue
                if line_ndx % self.args['world_size'] != \
                        self.args['global_rank']:
                    continue
                query = jobj['input'] if 'input' in jobj else jobj['query']
                passages = eval(jobj['passages'])
                positive_pids = self.inst_id2pos_pids[inst_id]
                if self.args['doc_match_weight'] > 0:
                    positive_dids = [
                        pid[:pid.index('::')] for pid in positive_pids
                    ]
                else:
                    positive_dids = None
                correctness = [
                    self.passage_correctness(p['pid'], positive_pids,
                                             positive_dids) for p in passages
                ]
                passages, correctness = self.limit_gpu_sequences(
                    passages, correctness, rand)
                logits1, prelogits1 = self.one_instance(query, passages)

                nll = -(
                    logits1.dot(torch.tensor(correctness).to(logits1.device)))
                loss_val = self.optimizer.step_loss(nll)
                self.loss_history.note_loss(loss_val)
                losses.append(loss_val)
                if not self.optimizer.should_continue():
                    break
            if self.epoch_now >= 0 and self.epoch_now < 10:
                model_path = os.path.join(self.output_dir,
                                          '_' + str(self.epoch_now))
                save_transformer(self.args, self.optimizer.model, self.tokenizer, save_dir=model_path)
            if losses:
                logger.info(
                    f'epoch: {self.epoch_now} \t total_loss: {sum(losses) / len(losses)} \t '
                    # f'ce_loss: {sum(losses_ce) / len(losses_ce)} \t kl_loss: {alpha * sum(losses_kl) / len(losses_kl)}'
                )

            meters = self.evaluate()
            total_score = sum([x for x in meters.values()])
            logger.info(f'meters is {meters}')
            if total_score >= best_score:
                best_score = total_score
                model_path = os.path.join(self.output_dir,
                                          'best_model.bin')
                save_transformer(self.args, self.optimizer.model, self.tokenizer, save_dir=model_path)
                logger.info(
                    'epoch %d obtain max score: %.4f, saving model to %s' %
                    (self.epoch_now, total_score, model_path))

        get_length = self.args['max_seq_length']
        logger.info(f'loss_history = {self.loss_history.loss_history}')
        logger.info(
            f'truncated to max length ({get_length}) {self.max_length_count} times'
        )
        save_transformer(self.args, self.optimizer.model, self.tokenizer)

    def evaluate(self, checkpoint_path=None, saving_fn=None, per_gpu_batch_size=32):
        """
        Evaluate testsets
        """
        args = {
            'output': './',
            'max_batch_size': 16,
            'exclude_instances': '',
            'include_passages': False,
            'do_lower_case': True,
            'max_seq_length': 512,
            'query_length': 195,
            'tokenizer_resize': True,
            'model_resize': True,
            'kilt_data': True
        }
        label_dict = getlabel(self.eval_dataset)
        mypreprocessor = self.preprocessor
        pipeline_ins = myDocumentGroundedDialogRerankPipeline(
            model=self.optimizer.model, preprocessor=mypreprocessor, **args)
        pipeline_ins(self.eval_dataset)
        ans = pipeline_ins.measure_result(label_dict)
        return ans

    def one_instance(self, query, passages):
        model = self.optimizer.model
        input_dict = {'query': query, 'passages': passages}
        inputs = self.preprocessor(input_dict)
        prelogits = model(inputs).logits
        logits = F.log_softmax(
            prelogits,
            dim=-1)[:, 1]  # log_softmax over the binary classification
        logprobs = F.log_softmax(
            logits, dim=0)  # log_softmax over the passages
        # we want the logits rather than the logprobs as the teacher labels
        return logprobs, prelogits

    def compute_kl_loss(self, p, q, pad_mask=None):

        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.)
            q_loss.masked_fill_(pad_mask, 0.)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss


def getlabel(eval_dataset):
    label_dict = []
    for data in eval_dataset['positive_pids']:
        label_dict.append(eval(data)[0])
    return label_dict


def block_shuffle(iter, *, block_size=20000, rand=random):
    assert block_size >= 4
    block = []
    for item in iter:
        block.append(item)
        if len(block) >= block_size:
            rand.shuffle(block)
            for _ in range(block_size // 2):
                yield block.pop(-1)
    rand.shuffle(block)
    for bi in block:
        yield bi


def save_transformer(hypers, model, tokenizer, *, save_dir=None):
    if hypers['global_rank'] == 0:
        if save_dir is None:
            save_dir = hypers['output_dir']
        # Create output directory if needed
        os.makedirs(save_dir, exist_ok=True)
        logger.info('Saving model checkpoint to %s', save_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (model.module if hasattr(model, 'module') else model
                         )  # Take care of distributed/parallel training
        torch.save(hypers, os.path.join(save_dir, 'training_args.bin'))
        model_to_save.save_pretrained(save_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)


def loaddataset(args_data):
    # 利用中文或者英文协同训练越南语和法语
    train_dataset = []
    evaluate_dataset = []

    if args_data.experiment_set == 'ori_single':
        dataset = MsDataset.load(
            f'DAMO_ConvAI/{args_data.lang}Doc2BotRerank',
            download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
        dataset_before_shuffer = [x for x in dataset]
        dataset_ori = dataset_before_shuffer[100:]
        random.shuffle(dataset_ori)
        train_dataset = dataset_ori[200:]
        evaluate_dataset = transfer_eval_dataset(dataset_ori[:200])
        logger.info(
            f'train_dataset range: {300}-{len(dataset_before_shuffer)}; evaluate_dataset range: {100}-{100 + 200}')
    elif args_data.experiment_set == 'ori_single_aug':
        with open(f'./co_dataset/{args_data.lang.lower()}_co_aug_dataset.jsonl') as f:
            aug_dataset = []
            for line in f.readlines():
                aug_dataset.append(json.loads(line))
        dataset_before_shuffer = [x for x in aug_dataset]
        if args_data.lang.lower() == 'vi':
            dataset_ori = dataset_before_shuffer[100:]
            random.shuffle(dataset_ori)
            train_dataset = dataset_ori[200:3446]
            evaluate_dataset = transfer_eval_dataset(dataset_ori[:200])
            logger.info(
                f'train_dataset range: {100 + 200}-{len(dataset_before_shuffer)}; evaluate_dataset range: {100}-{100 + 200}')
        if args_data.lang.lower() == 'fr':
            dataset_ori = dataset_before_shuffer[100:]
            random.shuffle(dataset_ori)
            train_dataset = dataset_ori[200:]
            evaluate_dataset = transfer_eval_dataset(dataset_ori[:200])
            logger.info(
                f'train_dataset range: {100 + 200}-{len(dataset_before_shuffer)}; evaluate_dataset range: {100}-{100 + 200}')

    elif args_data.experiment_set == 'ori_co_aug':
        with open(f'./co_dataset/{args_data.lang.lower()}_co_aug_dataset.jsonl') as f:
            co_aug_dataset = []
            for line in f.readlines():
                co_aug_dataset.append(json.loads(line))
        dataset_before_shuffer = [x for x in co_aug_dataset]
        dataset_ori = dataset_before_shuffer[100:3510] + dataset_before_shuffer[3610:]
        random.shuffle(dataset_ori)
        train_dataset = dataset_ori[200:3410] + dataset_ori[3610:]
        eval_dataset = dataset_ori[:200] + dataset_ori[3410:3610]
        evaluate_dataset = transfer_eval_dataset(eval_dataset)
        logger.info(
            f'train_dataset range: {300}-{3510}&&{3610 + 200}-{len(dataset_before_shuffer)} \n; '
            f'evaluate_dataset range: {100}-{300}&&{3610}-{3810}')
    elif args_data.experiment_set == 'ori_all_co_pretrain':
        with open(f'./co_dataset/Fr_Vi_En2Fr_En2Vi_reank.jsonl') as f:
            co_aug_dataset = []
            for line in f.readlines():
                co_aug_dataset.append(json.loads(line))
        dataset_before_shuffer = [x for x in co_aug_dataset]
        dataset_ori = dataset_before_shuffer[6955:]
        random.shuffle(dataset_ori)
        train_dataset = dataset_ori[500:-500]
        eval_dataset = dataset_ori[:500] + dataset_ori[-500:]
        evaluate_dataset = transfer_eval_dataset(eval_dataset)
        logger.info(f'train_dataset range: {500}-{len(dataset_before_shuffer) - 500} \n; '
                    f'evaluate_dataset range: {0}-{500}&&{len(dataset_before_shuffer) - 500}-{len(dataset_before_shuffer)}')
    elif args_data.experiment_set == 'ori_joint':
        train_dataset_vi = MsDataset.load('DAMO_ConvAI/ViDoc2BotRerank',
                                          download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                                          split='train')
        train_dataset_fr = MsDataset.load('DAMO_ConvAI/FrDoc2BotRerank',
                                          download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
                                          split='train')
        dataset_before_shuffer = [x for dataset in [train_dataset_fr, train_dataset_vi] for x in dataset]
        dataset_ori = dataset_before_shuffer
        random.shuffle(dataset_ori)
        train_dataset = dataset_ori[50:3510] + dataset_ori[3560:]
        eval_dataset = dataset_ori[:50] + dataset_ori[3510:3560]
        evaluate_dataset = transfer_eval_dataset(eval_dataset)
        logger.info(
            f'train_dataset range: {50}-{3510}&&{3560}-{len(dataset_before_shuffer)} \n; '
            f'evaluate_dataset range: {0}-{50}&&{3510}-{3560}')
    elif args_data.experiment_set == 'ori_joint_hard':
        with open(f'./co_dataset/fr_hard.jsonl') as f_in:
            train_dataset_fr = []
            f_csv = csv.DictReader(f_in)
            for row in f_csv:
                train_dataset_fr.append(row)
        with open(f'./co_dataset/vi_hard.jsonl') as f_in:
            train_dataset_vi = []
            f_csv = csv.DictReader(f_in)
            for row in f_csv:
                train_dataset_vi.append(row)
        dataset_before_shuffer = [x for dataset in [train_dataset_fr, train_dataset_vi] for x in dataset]
        dataset_ori = dataset_before_shuffer
        random.shuffle(dataset_ori)
        train_dataset = dataset_ori[50:3510] + dataset_ori[3560:]
        eval_dataset = dataset_ori[:50] + dataset_ori[3510:3560]
        evaluate_dataset = transfer_eval_dataset(eval_dataset)
        logger.info(
            f'train_dataset range: {50}-{3510}&&{3560}-{len(dataset_before_shuffer)} \n; '
            f'evaluate_dataset range: {0}-{50}&&{3510}-{3560}')
    else:
        train_dataset = []
        evaluate_dataset = []
        logger.info('No such experiment set!')

    return train_dataset, evaluate_dataset


def transfer_eval_dataset(eval_dataset):
    all_querys = []
    retrieval_result = []
    label_dict = []
    for data in eval_dataset:
        all_querys.append({"query": data["input"], 'response': '', 'positive': '', 'negative': ''})
        result = [p["text"] for p in eval(data["passages"])]
        label_dict.append(data['positive_pids'])
        retrieval_result.append(result)
    passage_to_id = {}
    ptr = -1
    for file_name in['fr', 'vi', 'en_fr', 'en_vi']:
        with open(f'./all_passages/{file_name}.json') as f:
            all_passages = json.load(f)
            for every_passage in all_passages:
                ptr += 1
                passage_to_id[every_passage] = str(ptr)

    input_list = []
    passages_list = []
    ids_list = []
    output_list = []
    positive_pids_list = []
    for label in label_dict:
        positive_pids_list.append(label)
    ptr = -1
    for x in tqdm(all_querys):
        ptr += 1
        now_id = str(ptr)
        now_input = x
        now_wikipedia = []
        now_passages = []
        all_candidates = retrieval_result[ptr]
        for every_passage in all_candidates:
            get_pid = passage_to_id[every_passage]
            now_wikipedia.append({'wikipedia_id': str(get_pid)})
            now_passages.append({"pid": str(get_pid), "title": "", "text": every_passage})
        now_output = [{'answer': '', 'provenance': now_wikipedia}]
        input_list.append(now_input['query'])
        passages_list.append(str(now_passages))
        ids_list.append(now_id)
        output_list.append(str(now_output))
    evaluate_dataset = {'input': input_list, 'id': ids_list, 'passages': passages_list, 'output': output_list,
                        'positive_pids': positive_pids_list}
    return evaluate_dataset


def main():
    ##########################################################
    # 数据集选择
    ##########################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', default='Vi', type=str, required=False,
                        help="language of the task")
    parser.add_argument('--experiment_set', default='ori_joint', type=str, required=False,
                        help="the setting of experiment")
    parser.add_argument('--save_path', default='./model/rerank/output_rerank_model', type=str, required=False,
                        help="the setting of experiment")
    parser.add_argument('--cuda_id', default='0', type=str, required=False,
                        help="the setting of experiment")
    parser.add_argument('--pretrain', default=False, type=bool, required=False,
                        help="whether pretrain in Zh and En")
    parser.add_argument('--pretrain_path', default='/root/autodl-tmp/output_rerank_choose_0328/best_model.bin', type=str, required=False,
                        help="Where the pretrained weights are located")
    parser.add_argument('--epoch', default=25, type=int, required=False,
                        help="rounds of training")
    parser.add_argument('--rdrop_alpha', default=1.5, type=float, required=False,
                        help="Rate of random drop loss")
    args_data = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args_data.cuda_id

    assert args_data.lang in ['Vi', 'Fr'], \
        "This language is not in the target language list"
    assert args_data.experiment_set in ['ori_joint', 'ori_single', 'ori_single_aug',
                                        'ori_co_aug', 'ori_all_co_aug_pretrain',
                                        'ori_all_co_pretrain','ori_joint_hard'], \
        "The experiment setting is not in the setting list"

    args = {
        'device': 'gpu',
        'tokenizer_name': '',
        'cache_dir': '',
        'instances_size': 1,
        'output_dir': args_data.save_path,
        'rdrop_alpha': args_data.rdrop_alpha,
        'max_num_seq_pairs_per_device': 32,
        'full_train_batch_size': 32,
        'gradient_accumulation_steps': 32,
        'per_gpu_train_batch_size': 1,
        'num_train_epochs': args_data.epoch,
        'train_instances': -1,
        'learning_rate': 2e-5,
        'max_seq_length': 512,
        'num_labels': 2,
        'fold': '',  # IofN
        'doc_match_weight': 0.0,  # 用于匹配文档的权重用的
        'query_length': 195,  # 195
        'resume_from': '',  # to resume training from a checkpoint
        'config_name': '',
        'do_lower_case': True,
        'weight_decay': 0.01,  # previous default was 0.01
        'adam_epsilon': 1e-8,
        'max_grad_norm': 1.0,
        'warmup_instances': 0.1,  # previous default was 0.1 of total
        'warmup_fraction': 0.0,  # only applies if warmup_instances <= 0
        'no_cuda': False,
        'n_gpu': 1,
        'seed': 42,
        'fp16': False,
        'fp16_opt_level': 'O1',  # previous default was O2
        'per_gpu_eval_batch_size': 8,
        'log_on_all_nodes': False,
        'world_size': 1,
        'global_rank': 0,
        'local_rank': -1,
        'tokenizer_resize': True,
        'model_resize': True
    }
    args[
        'gradient_accumulation_steps'] = args['full_train_batch_size'] // (
            args['per_gpu_train_batch_size'] * args['world_size'])

    print(f"Model pretrain is {args_data.pretrain}")

    train_dataset, eval_dataset = loaddataset(args_data)

    if args_data.pretrain == True:
        cache_path = args_data.pretrain_path
    else:
        # cache_path = 'DAMO_ConvAI/nlp_convai_ranking_pretrain'
        cache_path = './model/rerank/XLM_large'

    print(f"length of dataset is {len(train_dataset) + len(eval_dataset)}", args_data.experiment_set)
    trainer = myDocumentGroundedDialogRerankTrainer(
        model=cache_path, eval_dataset=eval_dataset,
        train_dataset=train_dataset, args=args)
    # 法语[0，3510]，越南语[3510,6955]
    trainer.train()


if __name__ == '__main__':
    main()
