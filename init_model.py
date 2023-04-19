import os

import torch
import torch.nn as nn

from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.models.nlp.dgds.backbone import DPRModel
from transformers import (XLMRobertaModel, XLMRobertaTokenizer)


class Wrapper(nn.Module):

    def __init__(self, encoder):
        super(Wrapper, self).__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, dummy_tensor):
        return self.encoder(input_ids, attention_mask).pooler_output


class MyDPRModel(nn.Module):
	def __init__(self):
		super().__init__()

		qry_encoder = XLMRobertaModel.from_pretrained('microsoft/infoxlm-large')
		ctx_encoder = XLMRobertaModel.from_pretrained('microsoft/infoxlm-large')
		qry_encoder.resize_token_embeddings(250007)
		ctx_encoder.resize_token_embeddings(250007)
		self.qry_encoder = Wrapper(qry_encoder)
		self.ctx_encoder = Wrapper(ctx_encoder)
		self.loss_fct = nn.CrossEntropyLoss()


model = MyDPRModel()
model_dir = './model/retrieval/nlp_convai_retrieval_pretrain_infoxlm_baseline_v1' 
torch.save(model.state_dict(), f'{model_dir}/pytorch_model.bin')

#### 测试初始化的模型是否可用
# model_dir = './model/retrieval'
# config = Config.from_file(
# 	os.path.join(model_dir, ModelFile.CONFIGURATION))
# model = DPRModel(model_dir, config)
# state_dict = torch.load(
# 	os.path.join(model_dir, ModelFile.TORCH_MODEL_BIN_FILE),
# 	map_location='cpu')
# model.load_state_dict(state_dict)
