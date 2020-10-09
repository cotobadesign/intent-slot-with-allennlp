#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn import util
from allennlp.predictors import Predictor


class IntentSlotEstimator(Model):

    def __init__(self, vocab, embedder, encoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels_slot = vocab.get_vocab_size('tags')
        self.classifier_slot = torch.nn.Linear(
            encoder.get_output_dim(), num_labels_slot)
        num_labels_intent = vocab.get_vocab_size('labels')
        self.classifier_intent = torch.nn.Linear(
            encoder.get_output_dim(), num_labels_intent)
        self.accuracy_slot = CategoricalAccuracy()
        self.accuracy_intent = CategoricalAccuracy()

    def get_metrics(self, reset=False):
        return {'slot accuracy': self.accuracy_slot.get_metric(reset),
                'intent accuracy': self.accuracy_intent.get_metric(reset)}

    def forward(self, text, label=None, tag=None):

        # 単語分散表現への変換

        embedded_text = self.embedder(text)

        # エンコーディング

        mask = util.get_text_field_mask(text)
        encoded_text = self.encoder(embedded_text, mask)

        # 文表現の取得

        lengths = util.get_lengths_from_binary_sequence_mask(mask)
        hidden_num = self.encoder.get_output_dim()
        if self.encoder.is_bidirectional():
            hidden_num = hidden_num//2
        encoded_tails = torch.cat(
            [encoded_text[b, lengths[b]-1, 0:hidden_num]
             for b in range(encoded_text.shape[0])]).view(-1, hidden_num)
        if self.encoder.is_bidirectional():
            encoded_tails = torch.cat(
                (encoded_tails, encoded_text[:, 0, hidden_num:]), dim=1)

        # クラス分類

        logits_slot = self.classifier_slot(encoded_text)
        probs_slot = torch.nn.functional.softmax(logits_slot, dim=-1)
        logits_intent = self.classifier_intent(encoded_tails)
        probs_intent = torch.nn.functional.softmax(logits_intent, dim=-1)
        output = {'probs_slot': probs_slot, 'probs_intent': probs_intent}

        # ロス計算

        if label is not None and tag is not None:
            self.accuracy_slot(logits_slot, tag, mask)
            self.accuracy_intent(logits_intent, label)
            loss = util.sequence_cross_entropy_with_logits(
                logits_slot, tag, mask)+torch.nn.functional.cross_entropy(
                    logits_intent, label)
            output['loss'] = loss

        return output


class IntentSlotEstimatorPredictor(Predictor):

    def predict(self, sentence):
        return self.predict_json({'sentence': sentence})

    def _json_to_instance(self, json_dict):
        sentence = json_dict['sentence']
        return self._dataset_reader.text_to_instance(sentence)
