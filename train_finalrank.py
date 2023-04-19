import os
import re
import string
from collections import Counter
import random
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import sacrebleu
import torch
import tqdm
from rouge import Rouge
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from modelscope.utils.logger import get_logger
from modelscope.trainers import EpochBasedTrainer
import transformers
import argparse
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer, XLMRobertaConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import csv

transformers.logging.set_verbosity_error()


logger = get_logger()

def collate(batch):
    query = [item['query'] for datas in batch for item in datas ]
    response = [item['response'] for datas in batch for item in datas]
    return query, response

def prepare_optimizer(model, lr, weight_decay, eps):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        'weight_decay':
            weight_decay,
    }, {
        'params': [
            p for n, p in model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        'weight_decay':
            0.0,
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
    return optimizer


def measure_result(result_dict):
    recall_k = [1, 2, 3, 4]
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


def prepare_scheduler(optimizer, epochs, steps_per_epoch, warmup_rate):
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    return scheduler


class Trainer(EpochBasedTrainer):
    def __init__(self, args = None, model_name="xlm-roberta-large", train_dataset=None, eval_dataset=None, **kwargs):
        access_token = "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"
        self.model_name = model_name
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        # self.model = XLMRobertaForSequenceClassification(self.model_name)
        self.config = XLMRobertaConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, additional_special_tokens=["<response>", "<user>", "<agent>", "<last_turn>"], use_auth_token=access_token)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, use_auth_token=access_token)
        # self.tokenizer = AutoModelForSequenceClassification.from_pretrained(self.model_name, additional_special_tokens=["<response>", "<user>", "<agnet>", "<last_turn>"])
        # self.tokenizer.add_special_tokens()
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = 'cuda' \
            if ('device' not in kwargs or kwargs['device'] == 'gpu') and torch.cuda.is_available() \
            else 'cpu'
        self.model.to(self.device)
        print("")


def train(args,
          trainer,
          total_epcoh=10,
          batch_size=1,
          accumulation_steps=1,
          learning_rate=1e-5,
          warmup_ratio=0.1,
          weight_decay=0.1,
          eps=1e-06,
          loss_log_freq=40,
          clip_grad_norm=1.0,
          checkpoint_path=None):
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path)
        trainer.model.load_state_dict(state_dict)

    model = trainer.model
    tokenizer = trainer.tokenizer
    device = trainer.device
    train_loader = DataLoader(
        dataset=trainer.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate
    )

    optimizer = prepare_optimizer(model, learning_rate, weight_decay, eps)
    steps_per_epoch = len(train_loader) // accumulation_steps
    scheduler = prepare_scheduler(optimizer, total_epcoh, steps_per_epoch, warmup_ratio)

    best_score = 0.0
    for epoch in range(total_epcoh):
        model.train()
        losses = []
        for index, payload in enumerate(tqdm.tqdm(train_loader)):
            query, response = payload
            query = [
                tokenizer.decode(
                    tokenizer([x], add_special_tokens=False, return_tensors='pt')['input_ids'][0][:args.max_query_len])
                for x in query
            ]
            generator_inputs = [
                ' '.join([' ', query[i], response[i] ])
                for i in range(len(query))
            ]
            input_ids = tokenizer.batch_encode_plus(
                list(generator_inputs), padding=True, return_tensors='pt').to(device)
            # label = torch.tensor([0 if i != 0 else 1 for i in range(args.topk)]).to(device)
            # label = label.expand(len(query)//args.topk, args.topk).reshape(-1)
            logits = model(input_ids=input_ids.input_ids, attention_mask=input_ids.attention_mask).logits
            logits = F.log_softmax(logits.view(-1, args.topk, 2), dim=-1)[:,:,0]
            gold_p = logits[:,0].view(-1,1)
            penalty = torch.div(1.0, (torch.sum(torch.pow(logits-gold_p, 2)).div(logits.shape[0])) + eps)
            label_mask = torch.tensor([0.0 if i != 0 else 1.0 for i in range(logits.shape[1])]).to(device)
            label_mask = label_mask.expand(logits.shape[0], logits.shape[1]).reshape(-1)
            logits = F.log_softmax(logits, dim=-1).view(-1)



            loss = -logits.dot(label_mask).div(label_mask.sum())

            if accumulation_steps > 1:
                loss = loss / accumulation_steps

            loss.backward()

            if (index + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            losses.append(loss.item())
            if (index + 1) % loss_log_freq == 0:
                logger.info(
                    f'epoch: {epoch} \t batch: {batch_size * index} \t loss: {sum(losses) / len(losses)} \t'
                )
                losses = []
        if losses:
            logger.info(
                f'epoch: {epoch} \t batch: last \t loss: {sum(losses) / len(losses)}'
            )

        meters = evaluate(args, trainer, batch_size=batch_size)
        score = meters['R@1']
        if score > best_score:
            best_score = score
            model_path = os.path.join(args.cache_dir, args.output_dir, 'finetuned_model.bin')

            torch.save(model.state_dict(), model_path)
            logger.info(
                'epoch %d obtain max score: %.4f, saving model to %s' %
                (epoch, score, model_path))


def evaluate(args, trainer, batch_size=4, checkpoint_path=None):
    model = trainer.model
    tokenizer = trainer.tokenizer
    device = trainer.device

    valid_loader = DataLoader(
        dataset=trainer.eval_dataset,
        batch_size=batch_size,
        collate_fn=collate)
    model.eval()

    with torch.no_grad():
        results = {
            'outputs': [],
            'targets': []
        }
        for index, payload in enumerate(tqdm.tqdm(valid_loader)):
            query, response = payload
            query = [
                tokenizer.decode(
                    tokenizer([x], add_special_tokens=False, return_tensors='pt')['input_ids'][0][:args.max_query_len])
                for x in query
            ]
            generator_inputs = [
                ' '.join(['<response>', response[i], query[i]])
                for i in range(len(query))
            ]
            input_ids = tokenizer.batch_encode_plus(
                list(generator_inputs), padding=True, return_tensors='pt').to(device)
            logits = model(input_ids=input_ids.input_ids, attention_mask=input_ids.attention_mask).logits
            logits = F.log_softmax(logits.view(-1, args.topk, 2), dim=-1)[:,:,0]
            top_k = logits.shape[1]
            # label_mask = torch.tensor([0.0 if i != 0 else 1.0 for i in range(top_k)]).to(device)
            # label_mask = label_mask.expand(logits.shape[0], logits.shape[1]).reshape(-1)
            logits = F.log_softmax(logits, dim=-1)
            choose_best = logits.argmax(dim=-1).view(-1)

            for i in range(len(response)//top_k):
                results['outputs'].append(response[i*top_k: (i+1)*top_k])
                results['targets'].append(response[choose_best[i] + i*top_k])
        meters = measure_result(results)
        output_dir = os.path.join(args.cache_dir, args.output_dir)
        # 如果output_dir不存在，则创建
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logger.info(meters)

        return meters




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default="0", type=str, required=False,
                        help="gpu id to use")
    parser.add_argument('--batch_size', default=16, type=int, required=False, help="batch size")
    parser.add_argument('--topk', default=5, type=int, required=False, help="topk")
    parser.add_argument('--epochs', default=20, type=int, required=False, help="number of epochs")
    parser.add_argument('--accumulation_steps', default=1, type=int, required=False, help="accumulation steps")
    parser.add_argument('--seed', default=42, type=int, required=False, help="random seed")
    parser.add_argument('--lr', default=2e-5, type=float, required=False, help="learning rate")
    parser.add_argument('--max_query_len', default=168, type=int, required=False, help="max query length")
    parser.add_argument('--cache_dir', default='./final_rerank', type=str, required=False,
                        help="cache directory")  # '/root/autodl-tmp/generation'  /root/autodl-tmp /workspace/chengs18/doc2dial/generation
    parser.add_argument('--model_name', default='joeddav/xlm-roberta-large-xnli', type=str, required=False,
                        help="model name")
    parser.add_argument('--output_dir', default="xlm-roberta-large-xnli", type=str, required=False,
                        help="output directory")
    args = parser.parse_args()
    # 打印args中参数
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    dataset = []
    raw_dataset = []
    golden_data = []
    with open('pretrain_data/hard_negative.json', 'r') as f:
        raw_dataset = json.load(f)
    with open('pretrain_data/corespond_data.jsonl', 'r') as f:
        for line in f:
            golden_data.append(json.loads(line)['response'])
    for i in range(len(raw_dataset)):
        data = raw_dataset[i]
        # 将golden_data[i]中的数据添加到data前面中
        data.insert(0, {
            "response": golden_data[i],
            "query": data[1]['query']
        })
        dataset.append(data)

    # 将数据集分为训练集和验证集, 并且将数据集打乱
    # dataset = dataset[:100]
    random.seed(42)
    random.shuffle(dataset)
    train_dataset = dataset[:int(len(dataset)*0.9)]
    eval_dataset = dataset[int(len(dataset)*0.9):]
    # for i in range(100):
    #     data = []
    #     for j in range(5):
    #         response = '<response>' + str(j)
    #         query = '<last_turn>' + str(j)
    #         data.append({
    #             'response': response,
    #             'query': query
    #         })
    #     train_dataset.append(data)
    # eval_dataset = train_dataset

    trainer = Trainer(args, model_name=args.model_name ,train_dataset=train_dataset, eval_dataset=eval_dataset)
    train(args, trainer, batch_size=args.batch_size,
          accumulation_steps=args.accumulation_steps, learning_rate=args.lr, total_epcoh=args.epochs)
    evaluate(args, trainer, batch_size=args.batch_size)

