import os
import json
import argparse

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.msdatasets import MsDataset
from modelscope.trainers.nlp.document_grounded_dialog_retrieval_trainer import \
    DocumentGroundedDialogRetrievalTrainer
from modelscope.utils.constant import DownloadMode

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_pretrain', default=False, type=bool, required=False, help="whether to do pretrain")
    parser.add_argument('--cache_path', default='./model/retrieval/nlp_convai_retrieval_pretrain_infoxlm_baseline_v1', type=str, required=False,
                        help="cache directory") 
    parser.add_argument('--total_epoches', default=60, type=int, required=False, help="number of epochs")
    parser.add_argument('--batch_size', default=128, type=int, required=False, help="batch size")
    parser.add_argument('--per_gpu_batch_size', default=256, type=int, required=False, help="batch size for evaluate")
    parser.add_argument('--accumulation_steps', default=1, type=int, required=False, help="accumulation steps")
    parser.add_argument('--lr', default=2e-5, type=float, required=False, help="learning rate")
    # parser.add_argument('--do_drop', default=False, type=bool, required=False, help="whether to use R-drop")

    args = parser.parse_args()

    all_passages = []
    for file_name in ['fr', 'vi']:
        with open(f'all_passages/{file_name}.json') as f:
            all_passages += json.load(f)

    cache_path = args.cache_path

    ### 原始数据集
    fr_train_dataset = MsDataset.load(
        'DAMO_ConvAI/FrDoc2BotRetrieval',
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

    vi_train_dataset = MsDataset.load(
        'DAMO_ConvAI/ViDoc2BotRetrieval',
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)

    origin_dataset = [x for dataset in [fr_train_dataset, vi_train_dataset] for x in dataset]

    train_dataset = origin_dataset[50:len(origin_dataset)-50]
    eval_dataset = origin_dataset[:50] + origin_dataset[len(origin_dataset)-50:]
    
    ### 预训练的数据集
    pretrain_dataset = []
    pretrain_data_file = ["en.jsonl", "zh.jsonl", "en_fr.jsonl", "en_vi.jsonl", "zh_fr.jsonl", "zh_vi.jsonl"]
    for data_file in pretrain_data_file:
        with open(f'./data_pretrain/retrieval/{data_file}') as f:
            pretrain_dataset += [json.loads(line) for line in f.readlines()]

    if args.do_pretrain:
        trainer = DocumentGroundedDialogRetrievalTrainer(
            model=cache_path,
            train_dataset=pretrain_dataset,
            eval_dataset=eval_dataset,
            all_passages=all_passages)
    else:
        trainer = DocumentGroundedDialogRetrievalTrainer(
            model=cache_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            all_passages=all_passages)
    trainer.train(
        batch_size=args.batch_size,
        per_gpu_batch_size=args.per_gpu_batch_size,
        total_epoches=args.total_epoches,
        accumulation_steps=args.accumulation_steps,
        learning_rate= args.lr)
    trainer.evaluate(
        per_gpu_batch_size=args.per_gpu_batch_size,
        checkpoint_path=os.path.join(trainer.model.model_dir,
                                    'finetuned_model.bin'))
