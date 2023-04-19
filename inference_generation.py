import os
import json
from modelscope.trainers.nlp.document_grounded_dialog_generate_trainer import \
    DocumentGroundedDialogGenerateTrainer
from train_generation import Trainer, evaluate
import argparse
import langid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default="0", type=str, required=False,
                        help="gpu id to use")
    parser.add_argument('--batch_size', default=32, type=int, required=False, help="batch size")
    parser.add_argument('--model_nums', default=1, type=int, required=False, help="number of models to infer, only "
                                                                                  "langs > 2 is valid")
    parser.add_argument('--cache_dir', default='./model', type=str, required=False,
                        help="cache directory")  # '../DAMO-ConvAI/acl23doc2dial'  /root/autodl-tmp
    parser.add_argument('--output_dir', default='generation', type=str,  # ViFr_0.05_drop_0.02
                        required=False,
                        help="pretrain model directory")
    parser.add_argument('--pretrain_model_dir', default='generation', type=str, # ViFr_0.05_drop_0.02
                        required=False,
                        help="pretrain model directory")
    parser.add_argument('--model_name', default='./model/generation/mbart-large-50', type=str, required=False,
                        help="model name used in huggingface")
    parser.add_argument('--ckpt_name', default='Vi_Fr_pretrain_on_Vi_Fr_En2Vi_En2Fr_finetuned_model.bin', type=str, required=False, help='model name to use') # Vi_Fr_pretrain_on_Vi_Fr_En2Vi_En2Fr_finetuned_model.bin
    parser.add_argument('--Fr_ckpt_name', default='Fr_pretrain_on_En_Zh_finetuned_model.bin', type=str, required=False, help='Fr model name to use')
    parser.add_argument('--Vi_ckpt_name', default='Vi_pretrain_on_En_Zh_finetuned_model.bin', type=str, required=False, help='Vi model name to use')
    parser.add_argument('--max_passage_len', default=384, type=int, required=False, help="max passage length")
    parser.add_argument('--max_query_len', default=96, type=int, required=False, help="max query length")
    parser.add_argument('--num_beams', default=3, type=int, required=False, help="num beams")
    parser.add_argument('--do_drop', default=False, type=bool, required=False, help="whether to use R-drop")
    parser.add_argument('--use_type_embeddings', default=False, type=bool, required=False,
                        help="whether to use type embeddings")
    parser.add_argument('--use_fid', default=False, type=bool, required=False, help="whether to use FID")
    parser.add_argument('--num_passages', default=2, type=int, required=False, help="number of passages")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    with open('all_passages/id_to_passage.json') as f:
        id_to_passage = json.load(f)

    eval_dataset = []
    eval_dataset_fr = []
    eval_dataset_vi = []
    index = []
    assert args.model_nums in [1, 2], \
        "This model number is not in the target model number list"
    with open('./model/rerank/rerank_output.jsonl') as f:
        for line in f.readlines():

            sample = json.loads(line)
            lang_type = langid.classify(sample['input'])[0]
            index.append(lang_type)
            if lang_type == 'fr':  # 识别语言为法语
                eval_dataset_fr.append({
                    'query': sample['input'],
                    'rerank': json.dumps([id_to_passage[x['wikipedia_id']] for x in sample['output'][0]['provenance']],
                                         ensure_ascii=False),
                    'response': '<response> @'
                })
            elif lang_type == 'vi':  # 识别语言为越南语
                eval_dataset_vi.append({
                    'query': sample['input'],
                    'rerank': json.dumps([id_to_passage[x['wikipedia_id']] for x in sample['output'][0]['provenance']],
                                         ensure_ascii=False),
                    'response': '<response> @'
                })
            eval_dataset.append({  # 两种语言都加入
                'query': sample['input'],
                'rerank': json.dumps([id_to_passage[x['wikipedia_id']] for x in sample['output'][0]['provenance']],
                                     ensure_ascii=False),
                'response': '<response> @'
            })
    # eval_dataset
    # cache_path = '../DAMO-ConvAI/acl23doc2dial/DAMO_ConvAI/nlp_convai_generation_pretrain'
    cache_path = os.path.join(args.cache_dir, args.pretrain_model_dir)
    if args.model_nums == 1: # 两种用一个模型推理
        trainer = Trainer(
            args,
            model_name= args.model_name,
            train_dataset=None,
            eval_dataset=eval_dataset,
        )
        ckpt_name = args.ckpt_name
        model_path = os.path.join(cache_path,
                                  ckpt_name)
        evaluate(args, trainer, checkpoint_path=model_path, batch_size=args.batch_size)

        with open(f'{cache_path}/evaluate_result.json') as f:
            predictions = json.load(f)['outputs']

        with open('model_outputStandardFileBaseline.json', 'w') as f:
            for query, prediction in zip(eval_dataset, predictions):
                f.write(json.dumps({
                    'query': query['query'],
                    'response': prediction.replace('<response>', '').strip()
                }, ensure_ascii=False) + '\n')

    elif args.model_nums == 2: # 两种语言用两个模型推理
        trainer_fr = Trainer(
            train_dataset=None,
            eval_dataset=eval_dataset_fr,
        )
        trainer_vi = Trainer(
            train_dataset=None,
            eval_dataset=eval_dataset_vi,
        )
        model_path_fr = os.path.join(cache_path,
                                     args.Fr_ckpt_name)
        model_path_vi = os.path.join(cache_path,
                                     args.Vi_ckpt_name)
        result_name_fr = 'Fr_evaluate_result.json'
        result_name_vi = 'Vi_evaluate_result.json'
        evaluate(args, trainer_fr, checkpoint_path=model_path_fr, result_name=result_name_fr)
        evaluate(args, trainer_vi, checkpoint_path=model_path_vi, result_name=result_name_vi)

        with open(f'{cache_path}/{result_name_fr}') as f:
            predictions_fr = json.load(f)['outputs']

        with open(f'{cache_path}/{result_name_vi}') as f:
            predictions_vi = json.load(f)['outputs']

        with open('2model_outputStandardFileBaseline.json', 'w') as f:
            idx_fr, idx_vi, idx = 0, 0, 0
            while idx_fr < len(eval_dataset_fr) and idx_vi < len(eval_dataset_vi):
                if index[idx] == 'fr':
                    f.write(json.dumps({
                        'query': eval_dataset_fr[idx_fr]['query'],
                        'response': predictions_fr[idx_fr].replace('<response>', '').strip()
                    }, ensure_ascii=False) + '\n')
                    idx_fr += 1
                else:
                    f.write(json.dumps({
                        'query': eval_dataset_vi[idx_vi]['query'],
                        'response': predictions_vi[idx_vi].replace('<response>', '').strip()
                    }, ensure_ascii=False) + '\n')
                    idx_vi += 1
                idx += 1
            while idx_fr < len(eval_dataset_fr):
                f.write(json.dumps({
                    'query': eval_dataset_fr[idx_fr]['query'],
                    'response': predictions_fr[idx_fr].replace('<response>', '').strip()
                }, ensure_ascii=False) + '\n')
                idx_fr += 1
                idx += 1
            while idx_vi < len(eval_dataset_vi):
                f.write(json.dumps({
                    'query': eval_dataset_vi[idx_vi]['query'],
                    'response': predictions_vi[idx_vi].replace('<response>', '').strip()
                }, ensure_ascii=False) + '\n')
                idx_vi += 1
                idx += 1

