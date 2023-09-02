import datetime

import torch
from django.db import models
from django.conf import settings
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List
from rest_framework.authtoken.models import Token
from django.db.models.signals import post_save
from django.dispatch import receiver
from torch.serialization import default_restore_location
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
import yaml

from .model.config import Config, ID2LABEL_NORM, LABEL2ID_NORM, ID2LABEL_STATUS
from .model.norm_detector import NormDectectorModel


# Create your models here.
@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_auth_token(sender, instance=None, created=False, **kwargs):
    if created:
        Token.objects.create(user=instance)

class NormDectector(models.Model):
  model = None

  def __init__(self):
    PROJECT_ROOT = getattr(settings, "PROJECT_ROOT", "")
    CONFIG_PATH = getattr(settings, 'CONFIG_PATH', f"{PROJECT_ROOT}/config/config.yaml")
    config = None
    with open(CONFIG_PATH, 'r') as file:
      config = yaml.safe_load(file)

    self.model_config = Config()
    if 'model_path' in config:
        self.model_config.model_path = f"{PROJECT_ROOT}/{config['model_path']}"

    if 'saved_model_file' in config:
        self.model_config.saved_model_file = f"{PROJECT_ROOT}/{config['saved_model_file']}"
    cpu = False
    if 'cpu' in config:
        cpu = config['cpu']
    if cpu:
        self.model_config.device = 'cpu'

    self.model = NormDectectorModel(self.model_config)
    print(f"Load model from {self.model_config.saved_model_file}")
    checkpoint = torch.load(self.model_config.saved_model_file,
                            map_location=lambda s, l: default_restore_location(s, "cpu"))
    self.model.load_state_dict(checkpoint["state_dict"])
    self.model.eval()
    self.tokenizer = BertTokenizer.from_pretrained(self.model_config.model_path)

    norm_list = LABEL2ID_NORM.keys()
    self.norm_list = [x for x in norm_list if x not in ["noann", "none"]]


  def _convert_ids_to_features(self, text):
      label = None
      temp  = self.tokenizer.encode_plus(text, label, padding='max_length',
                                   max_length=240)  # token_type_ids, input_ids, attention_mask

      input_ids = temp['input_ids']
      token_type_ids = temp['token_type_ids']
      attention_mask = temp['attention_mask']

      all_input_ids = torch.tensor([input_ids], dtype=torch.long)
      all_input_mask = torch.tensor([token_type_ids], dtype=torch.long)
      all_segment_ids = torch.tensor([attention_mask], dtype=torch.long)
      return all_input_ids, all_input_mask, all_segment_ids

  def _evaluate(self, features):
      input_ids = features[0].to(self.model_config.device)
      input_mask = features[1].to(self.model_config.device)
      segment_ids = features[2].to(self.model_config.device)

      with torch.no_grad():
          logits = self.model(input_ids, input_mask, segment_ids)

      pred_proba_norm = torch.nn.functional.softmax(logits['logits_norm'], dim=-1)
      predic_norm = torch.argmax(pred_proba_norm, -1).cpu().numpy()

      proba_norm = torch.sum(pred_proba_norm[:, 0:7], -1)
      proba_not_norm = torch.sum(pred_proba_norm[:, 7:], -1)
      ratio = proba_norm / proba_not_norm
      llr = torch.log(ratio)

      pred_proba_status = torch.nn.functional.softmax(logits['logits_status'], dim=-1)
      predic_status = torch.argmax(pred_proba_status, -1).cpu().numpy()

      pred_proba_positives = torch.nn.functional.softmax(pred_proba_status[:, 0:2], dim=-1)
      predic_status_positive = torch.argmax(pred_proba_positives, -1).cpu().numpy()

      # predict_norm_7_mask = np.where(predict_norm_all == 7, 1, 0)
      # predict_status_all[predict_norm_7_mask] = 3
      # predict_norm_8_mask = np.where(predict_norm_all == 8, 1, 0)
      # predict_status_all[predict_norm_8_mask] = 2

      predict_norm_label = ID2LABEL_NORM[predic_norm[0]]
      if predict_norm_label in self.norm_list:
          pred_status_positive = ID2LABEL_STATUS[predic_status_positive[0]]
      else:
          pred_status_positive = "EMPTY_NA"

      result = {"norm": predict_norm_label, "status": pred_status_positive,
                "llr": '%.2f' % llr[0]}
      print(result)
      return result["norm"], result["status"], result["llr"]

  @classmethod
  def load(cls):
      if cls.model is None:
          cls.model = NormDectector()
          print("load NormDectector model")
      return cls.model

  def detect_norm(self, in_message):
    if 'asr_text' in in_message:
        text = in_message['asr_text']
        features_data = self._convert_ids_to_features(text)
        results_ = self._evaluate(features_data)
        output = NormDetectionOutput(results_[0], results_[1], results_[2], in_message['uuid'], str(datetime.datetime.now().timestamp()))
        return output
    return NormDetectionOutput("none", "EMPTY_NA", 0.00, in_message['uuid'], str(datetime.datetime.now().timestamp()))

@dataclass_json
@dataclass
class CCUMessageInput:
  uuid: str
  asr_text: str
  asr_language: str
  asr_language_code: str
  timestamp: str

@dataclass_json
@dataclass
class RequestMessage:
  time_seconds: float
  queue: str
  message: CCUMessageInput

@dataclass_json
@dataclass
class NormDetectionOutput:
    name: str
    status: str
    llr: float
    trigger_id: List[str]
    timestamp: str

@dataclass_json
@dataclass
class ResponseMessage:
  time_seconds: float
  queue: str
  message: NormDetectionOutput
