# coding:utf-8
import falcon
import json
import io
import os
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
import numpy as np
from datetime import date
from PIL import Image
from falcon_multipart.middleware import MultipartMiddleware

class DebugResource:
    def on_post(self, req, res):
        """Handles POST requests"""
        data = req.get_param('file').file.read()
        pilimg = Image.open(io.BytesIO(data))
        x = np.asarray(pilimg)

        # 画像を読み込んで4次元テンソルへ変換
        x = np.expand_dims(x, axis=0)

        # 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要！
        # これを忘れると結果がおかしくなるので注意
        x = x / 255.0

        # クラスを予測
        # 入力は1枚の画像なので[0]のみ
        pred = model.predict(x)[0]

        # 予測確率が高いトップ5を出力
        top = 5
        top_indices = pred.argsort()[-top:][::-1]
        result = classes[top_indices[0]]

        # 予測確率が高いトップ1を出力
        res.status = falcon.HTTP_200
        res.body = json.dumps({'result':result})

# 画像認識のイニシャライズ
result_dir = 'results'

classes = ['Tulip', 'Snowdrop', 'LilyValley', 'Bluebell', 'Crocus',
           'Iris', 'Tigerlily', 'Daffodil', 'Fritillary', 'Sunflower',
           'Daisy', 'ColtsFoot', 'Dandelion', 'Cowslip', 'Buttercup',
           'Windflower', 'Pansy']
nb_classes = len(classes)

img_height, img_width = 150, 150
channels = 3

# VGG16
input_tensor = Input(shape=(img_height, img_width, channels))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

# FC
fc = Sequential()
fc.add(Flatten(input_shape=vgg16.output_shape[1:]))
fc.add(Dense(256, activation='relu'))
fc.add(Dropout(0.5))
fc.add(Dense(nb_classes, activation='softmax'))

# VGG16とFCを接続
model = Model(input=vgg16.input, output=fc(vgg16.output))

# 学習済みの重みをロード
model.load_weights(os.path.join(result_dir, 'finetuning.h5'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# WebAPIの起動
api = falcon.API(middleware=[MultipartMiddleware()])
api.add_route('/debug', DebugResource())