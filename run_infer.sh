cp ./model/retrieval/nlp_convai_retrieval_pretrain_infoxlm_baseline_v1/finetuned_model.bin ./model/retrieval/nlp_convai_retrieval_pretrain_infoxlm_baseline_v1/pytorch_model.bin

python inference_retrieval.py

python inference_rerank.py

python ensemble_rerank.py

python inference_generation.py