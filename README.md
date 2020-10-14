# intent-slot-with-allennlp

## 概要

2020年10月8日 18:00-20:00 に行われた「対話システム勉強会（第一回）OSSを利用した対話システムの作り方」の、「2. AllenNLP による自然言語理解機能の実装」で使用したコードです。

https://cotobaagent-developers-community.connpass.com/event/188047/

## 1. 学習データの準備

% python scripts/corpus.py

- 上記 scripts/corpus.py の形態素解析を jumannpp から janome に変更した版

% python scripts/corpus_janome.py

## 2. インテント認識モデルの推定とテスト

% python scripts/train_intent_estimator.py

## 3. スロット認識モデルの推定とテスト

% python scripts/train_slot_estimator.py

## 4. インテントとスロットの同時認識モデルの推定とテスト

% python scripts/train_intent_slot_estimator.py
