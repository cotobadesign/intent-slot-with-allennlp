#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pyknp import Jumanpp

# 半角全角変換の準備

zen = ''.join(chr(0xff01 + i) for i in range(94)) + '￥　'
han = ''.join(chr(0x21 + i) for i in range(94)) + '¥ '
han2zen = str.maketrans(han, zen)

# 形態素解析の準備

jumanpp = Jumanpp()

# 学習開発データの作成

with open('data/training_data_sample.json', 'r') as f:
    corpus = json.load(f)

for key in corpus:

    with open('data/%s' % key, 'w') as f:

        for data in corpus[key]:
            text = data['text'].translate(han2zen)

            # 形態素解析処理

            mrphs = [mrph.midasi
                     for mrph in jumanpp.analysis(text).mrph_list()]

            # 文字位置から単語位置への変換辞書の作成

            c2w = {}
            c = 0
            w = 0
            for mrph in mrphs:
                for i in range(len(mrph)):
                    c2w[c] = w
                    c += 1
                w += 1

            # スロット列の作成

            slots = ['O']*len(mrphs)

            for slot in data['slot']:
                type = slot['type']
                start = slot['start']
                end = slot['end']
                for c in range(start, end):
                    if slots[c2w[c]] != 'O':
                        continue
                    if c == start:
                        slots[c2w[c]] = 'B-'+type
                    else:
                        slots[c2w[c]] = 'I-'+type

            # 学習開発データの書き出し

            print(' '.join(
                ['%s:%s' % (ms[0], ms[1]) for ms in zip(mrphs, slots)]),
                  file=f, end=' ')
            print('<=> %s' % data['intent'][0], file=f)
