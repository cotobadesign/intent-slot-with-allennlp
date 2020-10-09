#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyknp import Jumanpp

from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.fields import TextField, LabelField, SequenceLabelField

class IntentSlotDatasetReader(DatasetReader):

    def __init__(self,
                 lazy=False,
                 max_tokens=64):
        super().__init__(lazy)
        self.token_indexers = {'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens
        self.jumanpp = Jumanpp()

    def _read(self, file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip().split()
                label = line[-1]
                line = [tt.split(':') for tt in line[:-2]]
                text = [Token(tt[0]) for tt in line][0: self.max_tokens]
                tags = [tt[1] for tt in line][0: self.max_tokens]
                yield self.text_to_instance(text, label, tags)

    def tokenizer(self, text):
        text = [Token(mrph.midasi)
                for mrph in self.jumanpp.analysis(
                    text).mrph_list()][0: self.max_tokens]
        return text

    def text_to_instance(self, text, label=None, tags=None):
        text_field = TextField(text, self.token_indexers)
        fields = {'text': text_field}
        if label:
            label_field = LabelField(label, label_namespace='labels')
            fields['label'] = label_field
        if tags:
            tags_field = SequenceLabelField(tags, text_field,
                                            label_namespace='tags')
            fields['tag'] = tags_field
        return Instance(fields)
