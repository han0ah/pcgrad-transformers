import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import RobertaPreTrainedModel,RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

from torch.nn import CrossEntropyLoss

class RobertaForMTL(RobertaPreTrainedModel):
    # Task 0 : PAWS 2-class
    # Task 1 : Klue-NLI 3-class
    _TASK_NUM_CLASS = [2,3]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.classifier_list = nn.ModuleList(
            [nn.Linear(config.hidden_size, self._TASK_NUM_CLASS[i]) for i in range(2)]
        )
        self.dropout=nn.Dropout(p=config.hidden_dropout_prob)

        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        task_id: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier_list[task_id](pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1,self._TASK_NUM_CLASS[task_id]), labels.view(-1))
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
    


         
        
