import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from torch.nn import CrossEntropyLoss


# import pdb
class NormDectectorModel(nn.Module):

    def __init__(self, cfg):
        super(NormDectectorModel, self).__init__()
        self.cfg = cfg
        self.num_labels = cfg.num_labels
        self.config = AutoConfig.from_pretrained(cfg.model_path)
        self.roberta = AutoModel.from_pretrained(cfg.model_path, config=self.config)
        if 'electra' in cfg.model_path:
            self.config.hidden_size = 1024
        else:
            self.config.hidden_size = 768
        self.norm = nn.Linear(self.config.hidden_size, self.num_labels)
        self.status = nn.Linear(self.config.hidden_size, 4)
        self.norm_loss_fn = CrossEntropyLoss()
        self.status_loss_fn = CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):

        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids)
        if 'electra' in self.cfg.model_path:
            # pdb.set_trace()
            pooler_output = output[0]
            pooler_output = torch.mean(pooler_output, dim=1)
        else:
            pooler_output = output[1]
        logits_norm = self.norm(pooler_output)
        logits_status = self.status(pooler_output)

        if labels is not None:
            loss_norm = self.norm_loss_fn(logits_norm.view(-1, self.num_labels), labels[:, 0].view(-1))
            loss_status = self.status_loss_fn(logits_status.view(-1, 4), labels[:, 1].view(-1))
            loss = loss_norm + loss_status
            return loss, {"logits_norm": logits_norm, "logits_status": logits_status}
        else:
            return {"logits_norm": logits_norm, "logits_status": logits_status}


