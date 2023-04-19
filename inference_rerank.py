import json
from typing import Union, Dict, Any

import torch
from modelscope.models import Model
from modelscope.models.nlp import DocumentGroundedDialogRerankModel
from modelscope.pipelines.nlp import DocumentGroundedDialogRerankPipeline
from modelscope.preprocessors.nlp import \
    DocumentGroundedDialogRerankPreprocessor
from tqdm import tqdm


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

    def save(self, addr):
        file_out = open(addr, 'w')
        for every_dict in self.guess:
            file_out.write(json.dumps(every_dict) + '\n')

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


def main():
    file_in = open('./input.jsonl', 'r')
    all_querys = []
    for every_query in file_in:
        all_querys.append(json.loads(every_query))
    passage_to_id = {}
    ptr = -1
    for file_name in ['fr', 'vi']:
        with open(f'./all_passages/{file_name}.json') as f:
            all_passages = json.load(f)
            for every_passage in all_passages:
                ptr += 1
                passage_to_id[every_passage] = str(ptr)

    file_in = open('./model/retrieval/nlp_convai_retrieval_pretrain_infoxlm_baseline_v1/evaluate_result.json', 'r')
    retrieval_result = json.load(file_in)['outputs']
    input_list = []
    passages_list = []
    ids_list = []
    output_list = []
    positive_pids_list = []
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
        positive_pids_list.append(str([]))
    evaluate_dataset = {'input': input_list, 'id': ids_list, 'passages': passages_list, 'output': output_list,
                        'positive_pids': positive_pids_list}
    model_ids = ['_14', '_10', '_11', '_12', '_13', '_15',
                '_16', '_17', '_18', '_19', '_21', ]
    for model_id in model_ids:
        model_dir = f'./model/rerank/output_rerank_model/{model_id}'
        model_configuration = {
            "framework": "pytorch",
            "task": "document-grounded-dialog-rerank",
            "model": {
                "type": "doc2bot"
            },
            "pipeline": {
                "type": "document-grounded-dialog-rerank"
            },
            "preprocessor": {
                "type": "document-grounded-dialog-rerank"
            }
        }
        file_out = open(f'{model_dir}/configuration.json', 'w')
        json.dump(model_configuration, file_out, indent=4)
        file_out.close()
        args = {
            'output': './',
            'max_batch_size': 64,
            'exclude_instances': '',
            'include_passages': False,
            'do_lower_case': True,
            'max_seq_length': 512,
            'query_length': 512,
            'tokenizer_resize': True,
            'model_resize': True,
            'kilt_data': True
        }
        model = Model.from_pretrained(model_dir, **args)
        mypreprocessor = DocumentGroundedDialogRerankPreprocessor(
            model.model_dir, **args)
        pipeline_ins = myDocumentGroundedDialogRerankPipeline(
            model=model, preprocessor=mypreprocessor, **args)
        pipeline_ins(evaluate_dataset)
        pipeline_ins.save(f'./model/rerank/output_rerank_result/rerank_output{model_id}.jsonl')


if __name__ == '__main__':
    main()
