import os
import re
import string
from collections import Counter
import random

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import sacrebleu
import torch
import tqdm
from rouge import Rouge
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
import transformers
import argparse

from modelscope.msdatasets import MsDataset
from modelscope.utils.constant import DownloadMode
from modelscope.trainers import EpochBasedTrainer
from modelscope.utils.logger import get_logger
from transformers import MBartForConditionalGeneration, MBartTokenizer
import torch.nn.functional as F
import csv
from MEmBart import FiDmBart
from mBart_embed import EmmBart
transformers.logging.set_verbosity_error()


logger = get_logger()


def collate(batch):
    query = [item['query'] for item in batch]
    # 判断item字典中是否包含rerank键，如果包含则返回对应的值，否则返回None

    context = [json.loads(item['rerank']) if 'rerank' in item else json.loads(json.dumps(eval(item['passages']))) for
               item in batch]
    label = [item['response'] for item in batch]
    return query, context, label


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


def prepare_scheduler(optimizer, epochs, steps_per_epoch, warmup_rate):
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    return scheduler


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def matching_evaluate(references, predictions):
    f1 = em = total = 0
    for ref_text, prediction in zip(references, predictions):
        total += 1
        ground_truths = [ref_text]
        f1 += metric_max_over_ground_truths(f1_score, prediction,
                                            ground_truths)
        em += metric_max_over_ground_truths(exact_match_score, prediction,
                                            ground_truths)
    f1 = 100.0 * f1 / total
    em = 100.0 * em / total

    return f1, em


def measure_result(result_dict):
    meters = dict()

    hypothesis_list = [
        x.replace('<extra_id_0>', '') for x in result_dict['outputs']
    ]
    hypothesis_list = [x if len(x) > 10 else 'placeholder' for x in hypothesis_list]
    reference_list = [
        x.replace('<response>', '') for x in result_dict['targets']
    ]
    instance_num = len(reference_list)

    # F1
    f1, em = matching_evaluate(reference_list, hypothesis_list)
    meters['f1'] = f1

    # SacreBleu
    bleu_score = [
        sacrebleu.sentence_bleu(hypothesis, [reference]).score
        for hypothesis, reference in zip(hypothesis_list, reference_list)
    ]
    bleu_score = sum(bleu_score) / instance_num
    meters['bleu'] = bleu_score

    # Rouge-L
    rouge_func = Rouge()
    rouge_score = [
        x['rouge-l']['f']
        for x in rouge_func.get_scores(hypothesis_list, reference_list)
    ]
    rouge_score = (sum(rouge_score) / instance_num) * 100
    meters['rouge'] = rouge_score

    return meters


def compute_kl_loss(p, q, pad_mask=None):

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


def do_tokenize(tokenizer, query, context, device):
    question = [x.split('<agent>')[0] for x in query]
    t_question = [tokenizer([x], add_special_tokens=False, return_tensors='pt')['input_ids'][0] for x in
                  question]
    t_question = [torch.cat([torch.tensor([250004]), x]) for x in t_question]
    question_type_id = [torch.zeros_like(x) for x in t_question]

    history = [x.split(y)[1] for x, y in zip(query, question)]
    t_history = [tokenizer([x], add_special_tokens=False, return_tensors='pt')['input_ids'][0] for x in history]
    history_type_id = [torch.ones_like(x) for x in t_history]

    t_context = [tokenizer([x], add_special_tokens=False, return_tensors='pt')['input_ids'][0] for x in context]
    t_context = [torch.cat([torch.tensor([6, 250058]), x, torch.tensor([2, ])]) for x in t_context]
    context_type_id = [torch.ones_like(x) * 2 for x in t_context]
    token_type_ids = [torch.concat([x, y, z], dim=0) for x, y, z in
                      zip(question_type_id, history_type_id, context_type_id)]
    max_shape = torch.Tensor([t.shape for t in token_type_ids]).max(0).values.int()
    token_type_ids_list = []
    input_ids_list = []
    attention_mask_list = []
    input_ids = [torch.cat([t_question[i], t_history[i], t_context[i]], dim=-1) for i in range(len(t_question))]
    for i, t in enumerate(token_type_ids):
        new_t = torch.zeros(max_shape)
        new_input = torch.zeros(max_shape)
        new_attention = torch.zeros(max_shape)
        new_input[:input_ids[i].shape[0]] = input_ids[i]
        new_attention[:input_ids[i].shape[0]] = torch.ones_like(input_ids[i])
        new_t[:t.shape[0]] = t

        token_type_ids_list.append(new_t)
        input_ids_list.append(new_input)
        attention_mask_list.append(new_attention)
    token_type_ids = torch.stack(token_type_ids_list, dim=0).long().to(device)
    input_ids = torch.stack(input_ids_list, dim=0).long().to(device)
    attention_mask = torch.stack(attention_mask_list, dim=0).long().to(device)
    return input_ids, token_type_ids, attention_mask


class Trainer(EpochBasedTrainer):
    def __init__(self, args = None, model_name='', **kwargs):
        """
        :param model:
        :param revision:
        :param args:
        :param kwargs:
        """
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mbart-large-50")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if args is not None and args.do_drop is True:
            # 设置模型dropout
            pass
            # self.model.config.dropout = args.dropout
            # 设置模型attention_dropout
            # self.model.config.attention_dropout = args.dropout
            # 设置模型activation_dropout
            # self.model.config.activation_dropout = args.dropout
            # 设置模型classifier_dropout
            # self.model.config.classifier_dropout = args.dropout

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name,
        #                                                additional_special_tokens=["<response>", "<user>", "<agnet>",
        #                                                                           "<last_turn>"])
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, additional_special_tokens=["<response>", "<user>", "<agent>", "<last_turn>", "<passage>"])
        self.model.resize_token_embeddings(len(self.tokenizer))
        if args is not None and args.use_fid:
            mbart = self.model
            self.model = FiDmBart(mbart.config)
            self.model.load_mbart(mbart.state_dict())
        if args is not None and args.use_type_embeddings:
            mbart = self.model
            self.model = EmmBart(mbart.config)
            self.model.load_mbart(mbart.state_dict())


        self.device = 'cuda' \
            if ('device' not in kwargs or kwargs['device'] == 'gpu') and torch.cuda.is_available() \
            else 'cpu'
        self.model.to(self.device)

        self.train_dataset = kwargs['train_dataset']
        self.eval_dataset = kwargs['eval_dataset']


def train(args,
          trainer,
          total_epoches=10,
          batch_size=16,
          accumulation_steps=1,
          learning_rate=1e-4,
          warmup_ratio=0.1,
          weight_decay=0.1,
          eps=1e-06,
          loss_log_freq=40,
          clip_grad_norm=1.0,
          checkpoint_path=None):
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path)
        if args.use_fid:
            trainer.model.load_mbart(state_dict)
        elif args.use_type_embeddings:
            trainer.model.load_emmbart(state_dict)
        else:
            trainer.model.load_state_dict(state_dict)

    model = trainer.model
    tokenizer = trainer.tokenizer
    device = trainer.device
    train_loader = DataLoader(
        dataset=trainer.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate)

    optimizer = prepare_optimizer(trainer.model, learning_rate,
                                  weight_decay, eps)
    steps_per_epoch = len(train_loader) // accumulation_steps
    scheduler = prepare_scheduler(optimizer, total_epoches,
                                  steps_per_epoch, warmup_ratio)
    best_score = 0.0
    for epoch in range(total_epoches):
        model.train()
        losses = []
        kl_losses = []
        ce_losses = []
        for index, payload in enumerate(tqdm.tqdm(train_loader)):
            query, context, label = payload
            now_batch = len(query)
            if args.use_fid:
                query = [
                    tokenizer.decode(
                        tokenizer([x], add_special_tokens=False, return_tensors='pt')['input_ids'][0][
                        :args.max_query_len])
                    for x in query for i in range(2)
                ]
                context = [
                    tokenizer.decode(
                        tokenizer([x[i]], add_special_tokens=False, return_tensors='pt')['input_ids'][0][
                        :args.max_passage_len])
                    for x in context for i in range(2)
                ]
            else:
                query = [
                    tokenizer.decode(
                        tokenizer([x], add_special_tokens=False, return_tensors='pt')['input_ids'][0][
                        :args.max_query_len])
                    for x in query
                ]
                context = [
                    tokenizer.decode(
                        tokenizer([x[0]], add_special_tokens=False, return_tensors='pt')['input_ids'][0][
                        :args.max_passage_len])
                    for x in context
                ]

            # context = [ # 引入随机噪声
            #     tokenizer.decode(
            #         tokenizer([x[0]], add_special_tokens=False, return_tensors='pt')['input_ids'][0][
            #         :args.max_passage_len]) if random.random() >= args.noisy_rate
            #     else tokenizer.decode(
            #         tokenizer([random.choice(x)], add_special_tokens=False, return_tensors='pt')['input_ids'][0][
            #         :args.max_passage_len])
            #     for x in context
            # ]

            if args.use_type_embeddings:
                input_ids, token_type_ids, attention_mask = do_tokenize(tokenizer, query, context, device)
            else:
                generator_inputs = [
                    ' '.join([query[i], '<passage>', context[i]])
                    for i in range(len(query))
                ]
                input_ids = tokenizer.batch_encode_plus(
                    list(generator_inputs), padding=True, return_tensors='pt').to(device)
                attention_mask = input_ids.attention_mask
                input_ids = input_ids.input_ids
            if args.use_fid:
                input_ids = input_ids.reshape(now_batch, args.num_passages, -1)
                attention_mask = attention_mask.reshape(now_batch, args.num_passages, -1)
            label_ids = tokenizer.batch_encode_plus(
                list(label), padding=True, return_tensors='pt').input_ids.to(device)
            if args.do_drop:
                if args.use_type_embeddings:
                    loss1, logits1 = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids,
                                           token_type_ids=token_type_ids)[0:2]
                    loss2, logits2 = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids,
                                           token_type_ids=token_type_ids)[0:2]
                else:
                    loss1, logits1 = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)[0:2]
                    loss2, logits2 = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)[0:2]

                kl_loss = compute_kl_loss(logits1, logits2) * args.kl_weight
                ce_loss = 0.5 * (loss1 + loss2)
                loss = ce_loss + kl_loss
                kl_losses.append(kl_loss.item())
                ce_losses.append(ce_loss.item())
            else:
                if args.use_type_embeddings:
                    loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids,
                                         token_type_ids=token_type_ids)[0:2]
                else:
                    loss, logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)[0:2]

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
                if args.do_drop is True:
                    logger.info(
                        f'epoch: {epoch} \t batch: {batch_size * index} \t loss: {sum(losses) / len(losses)} \t '
                        f'kl_loss: {sum(kl_losses) / len(kl_losses)} \t ce_loss: {sum(ce_losses) / len(ce_losses)}'
                    )
                else:
                    logger.info(
                        f'epoch: {epoch} \t batch: {batch_size * index} \t loss: {sum(losses) / len(losses)} \t'
                    )
                losses = []
                kl_losses = []
                ce_losses = []
        if losses:
            logger.info(
                f'epoch: {epoch} \t batch: last \t loss: {sum(losses) / len(losses)}'
            )

        meters = evaluate(args, trainer, batch_size=batch_size)
        total_score = sum([x for x in meters.values()])
        if args.save_early_model and 3 <= epoch <= 10:
            model_name = '_'.join(args.langs)
            if args.pretrain_on_langs is not None:
                model_name += '_pretrain_on_' + '_'.join(args.pretrain_on_langs)
            model_path = os.path.join(args.cache_dir, args.output_dir,
                                      model_name + '_early_' + str(epoch) + '.bin')
            state_dict = model.state_dict()
            torch.save(state_dict, model_path)
            logger.info(
                'epoch %d saving early model to %s' %
                (epoch, model_path))

        if total_score >= best_score:
            best_score = total_score
            model_name = '_'.join(args.langs)
            if args.pretrain_on_langs is not None:
                model_name += '_pretrain_on_' + '_'.join(args.pretrain_on_langs)
            model_path = os.path.join(args.cache_dir, args.output_dir,
                                      model_name + '_finetuned_model.bin')
            if args.use_fid:
                model.unwrap_encoder()
                state_dict = model.state_dict()
                torch.save(state_dict, model_path)
                model.wrap_encoder()
            else:
                state_dict = model.state_dict()
                torch.save(state_dict, model_path)

            logger.info(
                'epoch %d obtain max score: %.4f, saving model to %s' %
                (epoch, total_score, model_path))


def evaluate(args, trainer, batch_size=4, checkpoint_path=None, result_name='evaluate_result.json', num_return_sequences=1):
    model = trainer.model
    tokenizer = trainer.tokenizer
    device = trainer.device

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path)
        logger.info('load model from %s' % checkpoint_path)
        if args.use_fid:
            model.load_mbart(state_dict)
        elif args.use_type_embeddings:
            model.load_emmbart(state_dict)
        else:
            model.load_state_dict(state_dict)

    valid_loader = DataLoader(
        dataset=trainer.eval_dataset,
        batch_size=batch_size,
        collate_fn=collate)
    model.eval()
    with torch.no_grad():
        results = {'outputs': [], 'targets': []}
        for index, payload in enumerate(tqdm.tqdm(valid_loader)):
            query, context, label = payload
            now_batch = len(query)
            if args.use_fid:
                query = [
                    tokenizer.decode(
                        tokenizer([x], add_special_tokens=False, return_tensors='pt')['input_ids'][0][
                        :args.max_query_len])
                    for x in query for i in range(2)
                ]
                context = [
                    tokenizer.decode(
                        tokenizer([x[i]], add_special_tokens=False, return_tensors='pt')['input_ids'][0][
                        :args.max_passage_len])
                    for x in context for i in range(2)
                ]
            else:
                query = [
                    tokenizer.decode(
                        tokenizer([x], add_special_tokens=False, return_tensors='pt')['input_ids'][0][
                        :args.max_query_len])
                    for x in query
                ]
                context = [
                    tokenizer.decode(
                        tokenizer([x[0]], add_special_tokens=False, return_tensors='pt')['input_ids'][0][
                        :args.max_passage_len])
                    for x in context
                ]
            if args.use_type_embeddings:
                input_ids, token_type_ids, attention_mask = do_tokenize(tokenizer, query, context, device)
            else:
                generator_inputs = [
                    ' '.join([query[i], '<passage>', context[i]])
                    for i in range(len(query))
                ]
                input_ids = tokenizer.batch_encode_plus(
                    list(generator_inputs), padding=True, return_tensors='pt').to(device)
                attention_mask = input_ids.attention_mask
                input_ids = input_ids.input_ids
            if args.use_fid:
                input_ids = input_ids.reshape(now_batch, args.num_passages, -1)
                attention_mask = attention_mask.reshape(now_batch, args.num_passages, -1)

            if args.use_type_embeddings:
                outputs = model.generate(input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids,
                                         num_beams=args.num_beams,
                                         max_length=128,
                                         early_stopping=False,
                                         no_repeat_ngram_size=5,
                                         num_return_sequences=num_return_sequences,
                                         length_penalty=1,
                                         temperature=1)
            else:
                # outputs = model.generate(input_ids, num_beams=3, max_length=128, early_stopping=True,
                #                          no_repeat_ngram_size=5)
                outputs = model.generate(input_ids,
                                         attention_mask=attention_mask,
                                         num_beams=args.num_beams,
                                         max_length=128,
                                         early_stopping=False,
                                         no_repeat_ngram_size=5,
                                         num_return_sequences=num_return_sequences,
                                         length_penalty=1,
                                         temperature=1)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            label = tokenizer.batch_decode(
                tokenizer.batch_encode_plus(
                    label, add_special_tokens=False).input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False)

            results['outputs'] += predictions
            results['targets'] += label
        if num_return_sequences == 1:
            meters = measure_result(results)
            logger.info(meters)
        else:
            meters = None
        output_dir = os.path.join(args.cache_dir, args.output_dir)
        # 如果output_dir不存在，则创建
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        result_path = os.path.join(output_dir, result_name)
        with open(result_path, 'w') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    return meters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default="0", type=str, required=False,
                        help="gpu id to use")
    parser.add_argument('--langs', default=['Vi', 'Fr', ], type=str, nargs='+', required=False,
                        help="language of the task")#['Vi', 'Fr', 'En2Vi', 'En2Fr', 'Cn2Vi', 'Cn2Fr']
    parser.add_argument('--cache_dir', default='./model', type=str, required=False,
                        help="cache directory") # '/root/autodl-tmp/generation'  /root/autodl-tmp /workspace/chengs18/doc2dial/generation
    parser.add_argument('--pretrain_model_dir', default='generation', type=str,required=False,
                        help="pretrain model directory")
    parser.add_argument('--output_dir', default='generation', type=str, required=False, help="output directory")
    parser.add_argument('--model_name', default='./model/generation/mbart-large-50', type=str, required=False,
                        help="model name used in huggingface")
    parser.add_argument('--pretrain_on_langs', default=['Vi', 'Fr', 'En2Vi', 'En2Fr'], type=str, nargs='+', required=False,
                        help="pretrain model name")  #  ['En', 'Zh']  ['Vi', 'Fr', 'En', 'Zh'] ['Vi', 'Fr', 'En2Vi', 'En2Fr'] ['En', 'Zh', 'En2Vi', 'En2Fr', 'Cn2Vi', 'Cn2Fr']
    parser.add_argument('--epochs', default=20, type=int, required=False, help="number of epochs")
    parser.add_argument('--batch_size', default=16, type=int, required=False, help="batch size")
    parser.add_argument('--accumulation_steps', default=1, type=int, required=False, help="accumulation steps")
    parser.add_argument('--seed', default=42, type=int, required=False, help="random seed")
    parser.add_argument('--lr', default=1e-4, type=float, required=False, help="learning rate")
    parser.add_argument('--max_passage_len', default=300, type=int, required=False, help="max passage length")
    parser.add_argument('--max_query_len', default=128, type=int, required=False, help="max query length")
    parser.add_argument('--num_beams', default=3, type=int, required=False, help="num beams")
    parser.add_argument('--kl_weight', default=0.02, type=float, required=False, help="kl loss weight for R-drop")

    parser.add_argument('--test_data_rate', default=0.1, type=float, required=False, help="test data rate")
    parser.add_argument('--dropout', default=0.1, type=float, required=False, help="dropout rate")
    parser.add_argument('--do_pretrain', default=False, type=bool, required=False, help="whether to do pretrain")
    parser.add_argument('--noisy_rate', default=0, type=float, required=False, help="noisy rate for pretrain")
    parser.add_argument('--do_drop', default=True, type=bool, required=False, help="whether to use R-drop")
    parser.add_argument('--use_type_embeddings', default=False, type=bool, required=False, help="whether to use type embeddings")
    parser.add_argument('--use_fid', default=False, type=bool, required=False, help="whether to use FID")
    parser.add_argument('--num_passages', default=2, type=int, required=False, help="number of passages")
    parser.add_argument('--save_early_model', default=False, type=bool, required=False, help="whether to save early model")


    args = parser.parse_args()
    # 打印args中参数
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    train_dataset = []
    test_dataset = []

    for lang in args.langs:
        assert lang in ['Fr', 'Vi', 'En', 'Zh', 'En2Fr', 'En2Vi','Cn2Vi', 'Cn2Fr'], \
            "This language is not in the target language list"
        if lang in ['En', 'Zh']:
            ds = MsDataset.load(
                f'DAMO_ConvAI/{lang}Doc2BotDialogue',
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
        elif lang in ['Fr', 'Vi']:
            ds = MsDataset.load(
                f'DAMO_ConvAI/{lang}Doc2BotGeneration',
                download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
        elif lang in ['En2Fr', 'En2Vi', 'Cn2Vi', 'Cn2Fr']:
            with open(f'./pretrain_data/{lang}.csv') as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    ds.append({
                        'query': row[0],
                        'passages': row[1],
                        'response': row[2]
                    })

        ds = [x for x in ds]
        # ds = ds[:100]
        # 随机种子42 shuffle ds
        random.seed(args.seed)
        random.shuffle(ds)
        if lang in ['Fr', 'Vi'] and args.do_pretrain is True:
            for x in ds[int(len(ds)*0.20):int(len(ds)*0.40)]:
                test_dataset.append(x)
            ds = ds[int(len(ds)*0.40):] # 取前33%的数据作为finetune验证集

        for x in ds[0:int(len(ds)*args.test_data_rate)]:
            test_dataset.append(x)
        # for x in ds[0:int(len(ds)*args.test_data_rate/2)]:
        #     test_dataset.append(x)
        # for x in ds[int(len(ds)*(1-args.test_data_rate/2)):]:
        #     test_dataset.append(x)
        for x in ds[int(len(ds)*args.test_data_rate):]:
            train_dataset.append(x)
        ds = []

    with open('all_passages/id_to_passage.json') as f:
        id_to_passage = json.load(f)

    # cache_path = snapshot_download(args.pretrain_model_dir, cache_dir=args.cache_dir)
    trainer = Trainer(
        args=args,
        model_name=args.model_name,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    if args.pretrain_on_langs is not None:
        checkpoint_name = '_'.join(args.pretrain_on_langs) + '_finetuned_model.bin'
        checkpoint_path = os.path.join(args.cache_dir, args.pretrain_model_dir, checkpoint_name)
        logger.info(f'Load checkpoint from {checkpoint_path} ...')
    else:
        checkpoint_path = None
    train(args, trainer, batch_size=args.batch_size, accumulation_steps=args.accumulation_steps,
          total_epoches=args.epochs,
          learning_rate=args.lr, checkpoint_path=checkpoint_path)
    model_name = '_'.join(args.langs)
    if args.pretrain_on_langs is not None:
        model_name += '_pretrain_on_' + '_'.join(args.pretrain_on_langs)
    model_path = os.path.join(args.cache_dir, args.output_dir,
                              model_name + '_finetuned_model.bin')
    evaluate(args, trainer, checkpoint_path=model_path, batch_size=args.batch_size)
