import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch import nn, Tensor
from transformers import AutoModel, BertPreTrainedModel, PretrainedConfig, BertModel
from transformers.file_utils import ModelOutput
import numpy as np

logger = logging.getLogger(__name__)

        
@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None


class TextEncoderModel(BertPreTrainedModel):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 config: PretrainedConfig = None,
                 model_name_or_path: str = None,
                 normlized: bool = True,
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0,
                 ):
        super().__init__(config)

        if model_name_or_path:
            self.bert = AutoModel.from_pretrained(model_name_or_path)
        else:
            self.bert = BertModel(config)

        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

        self.normlized = normlized
        self.temperature = temperature
        self.config = config

        if not normlized:
            self.temperature = 1.0
            logger.info("reset temperature = 1.0 due to using inner product to compute similarity")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.bert.gradient_checkpointing_enable(**kwargs)

    def encode(self, features):
        if features is None:
            return None
        
        tmp_out = self.bert(**features, return_dict=True)
        reps_cls = tmp_out.last_hidden_state[:, 0]
        if self.normlized:
            reps_cls = torch.nn.functional.normalize(reps_cls, dim=-1)

        return reps_cls.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))
    
    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):

        q_reps_cls = self.encode(query)
        p_reps_cls = self.encode(passage)

        if self.negatives_cross_device and self.training:
            q_reps_cls = self._dist_gather_tensor(q_reps_cls)
            p_reps_cls = self._dist_gather_tensor(p_reps_cls)
        
        loss_cls = self.compute_loss(q_reps_cls, p_reps_cls)
        p_reps_cls = p_reps_cls.view(q_reps_cls.size(0), -1, q_reps_cls.size(1))

        return EncoderOutput(loss=loss_cls, q_reps=q_reps_cls, p_reps=p_reps_cls)

    def focal_loss(self, loss):
        gamma = 2 
        pt = torch.exp(-loss)
        focal_loss = (1 - pt) ** gamma * loss
        return focal_loss.mean()
    
    def compute_loss(self, q_reps, p_reps):
        scores = self.compute_similarity(q_reps, p_reps)
        scores = scores / self.temperature
        scores = scores.view(q_reps.size(0), -1)
        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (p_reps.size(0) // q_reps.size(0))

        loss_ce = self.cross_entropy(scores, target)
        loss = self.focal_loss(loss_ce)        
        return loss

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def save(self, output_dir: str):
        pretrained_output_dir = output_dir
        self.save_pretrained(pretrained_output_dir)
