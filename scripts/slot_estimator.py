#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn import util
from allennlp.predictors import Predictor


class SlotEstimator(Model):

    def __init__(self, vocab, embedder, encoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size('tags')
        self.classifier = torch.nn.Linear(
            encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}

    def forward(self, text, label=None, tag=None):

        # 単語分散表現への変換

        embedded_text = self.embedder(text)

        # エンコーディング

        mask = util.get_text_field_mask(text)
        encoded_text = self.encoder(embedded_text, mask)

        # クラス分類

        logits = self.classifier(encoded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output = {'probs': probs}

        # ロス計算

        if tag is not None:
            self.accuracy(logits, tag, mask)
            loss = util.sequence_cross_entropy_with_logits(logits, tag, mask)
            output['loss'] = loss

        return output


class SlotEstimatorPredictor(Predictor):

    def predict(self, sentence):
        return self.predict_json({'sentence': sentence})

    def _json_to_instance(self, json_dict):
        sentence = json_dict['sentence']
        return self._dataset_reader.text_to_instance(sentence)
