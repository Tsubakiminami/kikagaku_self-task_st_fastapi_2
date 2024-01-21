# キカガク　DXを推進するAI・データサイエンス人材育成コース　給付金要件　自走期間課題の提出
# 設定課題 XAI

# 画像をMINIST仕様画像に変換するなどの画像変換を行うモジュール

# アップロード画像をMNIST仕様画像に変換して戻す。
# V1.0.0  2023/11/30
# MNIST画像仕様
# 1.切り取った画像をグレースケールに変換
# 2.グレースケールにした画像を二値化して白黒にする
# 3.白黒画像を反転し、黒背景にする
# 4.ガウスブラーをかけて補完する
# 5.28x28pxの画像にサイズ変換する
# 6.numpy配列に変換する
#
# Streamlitのfile_uploaderでアップロードされたファイルは、
# 一般的にPIL（Python Imaging Library）を使用して読み込むことができます。
#

# 必要なモジュールをインポートする
from PIL import Image, ImageFilter, ImageOps
# import cv2
import numpy as np


def CvToMNIST_img(uploaded_file):
    # # streamlit形式(PIL)をopencv(Numpy配列かスカラー)に変換
    uploaded_image = Image.open(uploaded_file)
    # freehand_image = np.array(uploaded_image)

    # # グレースケール画像に変換
    # im = cv2.cvtColor(freehand_image, cv2.COLOR_BGR2GRAY)
    # # グレースケール画像を二値化
    # _, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
    # # 二値化された画像を反転
    # thresh = cv2.bitwise_not(thresh)
    # # ガウスブラーを適用して補完
    # thresh = cv2.GaussianBlur(thresh, (9, 9), 0)
    # # 画像データをリサイズ
    # im_t = cv2.resize(thresh, (28, 28), interpolation=cv2.INTER_CUBIC)
    # # OpenCVの画像をPIL形式に変換して戻す
    # return  Image.fromarray(im_t)

    # グレースケールに変換
    im = ImageOps.grayscale(uploaded_image)

    # グレースケール画像を二値化
    thresh = im.point(lambda x: 0 if x < 128 else 255, '1') # 128は閾値 閾値は適時調整

    # 二値化された画像を反転
    thresh = ImageOps.invert(thresh)

    # RGBモードに変換
    thresh_rgb = thresh.convert('RGB')
    # # ガウシアンブラーを適用
    # thresh_blurred = thresh_rgb.filter(ImageFilter.GaussianBlur(radius=8))
    thresh_blurred = thresh_rgb
    # グレースケールに戻す
    im_t = thresh_blurred.convert('L')

    # 画像データをリサイズ
    im_t_resized = im_t.resize((28, 28), resample=Image.BICUBIC)
    im_t_resized.save('./temp_img/thresh28x28.png')

    # # PillowのImageをNumPy配列に変換
    # Xt = np.array(im_t_resized)

    # # 正規化
    # Xt_std = Xt / 255.0

    # return Xt_std
    return im_t_resized    