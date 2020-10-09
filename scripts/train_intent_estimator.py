#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tempfile
import numpy as np
import torch

from allennlp.data import Vocabulary, PyTorchDataLoader
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.seq2vec_encoders import LstmSeq2VecEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.trainer import GradientDescentTrainer

from dataset_reader import IntentSlotDatasetReader
from intent_estimator import IntentEstimator, IntentEstimatorPredictor


if __name__ == '__main__':

    # 乱数の初期化

    seed = 777
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # 学習データの読み込み

    dataset_reader = IntentSlotDatasetReader()

    train_data = dataset_reader.read('data/training')
    valid_data = dataset_reader.read('data/validation')

    vocab = Vocabulary.from_instances(train_data+valid_data)
    vocab.save_to_files('vocab')

    train_data.index_with(vocab)
    valid_data.index_with(vocab)
    
    train_loader = PyTorchDataLoader(train_data, batch_size=8, shuffle=True)
    valid_loader = PyTorchDataLoader(valid_data, batch_size=8, shuffle=False)

    # モデルの作成

    embedder = BasicTextFieldEmbedder(
        {'tokens': Embedding(
            embedding_dim=10,
            num_embeddings=vocab.get_vocab_size('tokens'))})

    encoder = LstmSeq2VecEncoder(10, 32, bidirectional=True)
    # encoder = BagOfEmbeddingsEncoder(embedding_dim=10)

    model = IntentEstimator(vocab, embedder, encoder)
    model.cuda()

    # モデルの学習

    with tempfile.TemporaryDirectory() as serialization_dir:
        parameters = [
            [n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = AdamOptimizer(parameters)
        trainer = GradientDescentTrainer(
            model=model,
            serialization_dir=serialization_dir,
            data_loader=train_loader,
            validation_data_loader=valid_loader,
            num_epochs=20,
            optimizer=optimizer,
            cuda_device=0)

        trainer.train()

    # モデルの実行

    predictor = IntentEstimatorPredictor(model, dataset_reader)

    text = dataset_reader.tokenizer('東京駅から富山駅まで行きたいです')
    output = predictor.predict(text)
    print(text)
    print([(vocab.get_token_from_index(label_id, 'labels'), prob)
           for label_id, prob in enumerate(output['probs'])])

    text = dataset_reader.tokenizer('富山の天気はどうかな')
    output = predictor.predict(text)
    print(text)
    print([(vocab.get_token_from_index(label_id, 'labels'), prob)
           for label_id, prob in enumerate(output['probs'])])

    text = dataset_reader.tokenizer('富山の天気は関係なく旅行に行くよ')
    output = predictor.predict(text)
    print(text)
    print([(vocab.get_token_from_index(label_id, 'labels'), prob)
           for label_id, prob in enumerate(output['probs'])])
