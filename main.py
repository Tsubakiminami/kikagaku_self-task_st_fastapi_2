'''
キカガク DXを推進するAI・データサイエンス人材育成コース 給付金要件 自走期間課題の提出
 設定課題 XAI

 このアプリケーションは、自分で書いた数字１文字をスマホで撮影してアップロードすることで書かれた数字を判定します。
 2023/12/2 金曜日 大安

VS CODE 設定メモ：
 1. streamlitなどのモジュール名下部に could not resolved (黄色波線)が出る。
    コードは実行できるのでそのままでもいいが、仮想環境毎のモジュールへのパスを設定すると消せる。
    モジュールの場所を見つける方法：
 (venv_Streamlit) ik@mini 17_自走期間課題API % python
 Python 3.10.13 (main, Nov 30 2023, 22:13:08) [Clang 15.0.0 (clang-1500.0.40.1)] on darwin
 Type "help", "copyright", "credits" or "license" for more information.
 >>> import streamlit
 >>> print(streamlit.__file__)
 /Users/ik/Downloads/virtual_env/venv_Streamlit/lib/python3.10/site-packages/streamlit/__init__.py
 >>> 
 (ctr+D でpython終了)
 /Users/ik/Downloads/virtual_env/venv_Streamlit/lib/python3.10/site-packages/ がパス
    VS CODEの設定を開く方法：
    メニューから Code -> 基本設定 -> 設定　を選び設定画面を表示する。
    extra paths で検索する
    Python › Analysis: Extra Paths 下部の「項目の追加」ボタンをクリックする
     /Users/ik/Downloads/virtual_env/venv_Streamlit/lib/python3.10/site-packages/　を入力してOKをクリック
    一息すると黄色波線が消えた

その他設定メモ：
 1.コードの実行方法：
   ik@mini 17_自走期間課題API % source ../venv_Streamlit/bin/activate     
   (venv_Streamlit) ik@mini 17_自走期間課題API % streamlit run main.py   
 2.コードの停止方法：
   ctr+c で止まる

給付金要件について
 FastAPIでAPI(ここではバックエンドに相当する)を作成することで達成となる。
 このために、Streamlitですべてを作成したコードを、フロントエンドをStreamlitで作成しバックエンドをFastAPIで作成する。
 "DXを推進するAI・データサイエンス人材育成コース"のFastAPI実践で例示するコードを提出することでも要件達成とみなす措置だが
 すでにStreamlitでAPIを作成してしまったので、フロントエンド部分とバックエンド部分に分けて提出することにした。
 2024/01/02

main.py
 このコードでは、FastAPIを使用しトレーニング済みのMNISTモデルを提供するバックエンド（APIサーバ）を提供します。
 相手側のリクエストボディは、画像データをPOSTリクエストしてこのサーバに送信し、モデルによる数値の予測結果とXAI画像を返信します。

主な機能：
  1. FastAPIを使用したバックエンドサーバを提供する
  2. トレーニング済みのCNNモデルで画像からの予測結果を導く
  3. 予測結果の根拠となるXAIヒートマップ画像を生成する
  4. 各結果をJSON形式でクライアント側に返信する

使用方法：
  1. このコードでAPIサーバを起動
     python main.py
  2. 相手側クライアントからのHTTP POSTリクエストで画像データを送信

FastAPI:
    POST: データの作成
    GET: データの読み取り
    PUT: データの更新
    DELETE: データの削除

起動:
 ik@mini 17_自走期間課題FastAPI % source ../venv_Streamlit/bin/activate
(venv_Streamlit) ik@mini 17_自走期間課題FastAPI % uvicorn main:app --reload
INFO:     Will watch for changes in these directories: ['/Users/ik/Downloads/virtual_env/17_自走期間課題FastAPI']
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [3279] using StatReload
This main.py code for FastAPI starts. 2024-01-03 10:31:41.884467
This main.py code for FastAPI ends. 2024-01-03 10:31:45.570296
INFO:     Started server process [3281]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
^CINFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [3281]
INFO:     Stopping reloader process [3279]
(venv_Streamlit) ik@mini 17_自走期間課題FastAPI % 

(venv_Streamlit) ik@mini 17_自走期間課題FastAPI % pip install python-multipart
Collecting python-multipart
  Using cached python_multipart-0.0.6-py3-none-any.whl (45 kB)
Installing collected packages: python-multipart
Successfully installed python-multipart-0.0.6

'''

# 現在の時刻を取得
from datetime import datetime   # pip install datetime
# Stremlit とコード実行順序の関連を追跡し理解するためのPRINT文（ターミナルに出力される）

print(datetime.now(),'This main.py code for FastAPI starts.')

# 必要なモジュールをインポートする
from fastapi import FastAPI, File, UploadFile, Response
from PIL import Image
import io
import os

# FastAPIのインスタンスを作る
app = FastAPI()

# 自作モジュールにアクセスできるようにする
import my1_cvtmnist as my_cvtmnist
import my2_cnn as my_cnn
import my3_predictX as my_predict

# 作業フォルダ作成
dir_temp_img = 'temp_img'
if not os.path.exists(dir_temp_img):
  os.makedirs(dir_temp_img)

# ルートでの返信
@app.get('/')
async def read_root():
    return {'message' : 'キカガク API作成課題 2024/01/02'}  # JSONレスポンス  {} は辞書（dictionary）


# MNIST仕様画像に変換し数値を予想する
@app.post('/process_image_prediction/')
async def process_image_prediction(file: UploadFile = File(...)):
    # アップロードされた画像をPIL Imageオブジェクトに変換
    contents = await file.read()
    pil_image = Image.open(io.BytesIO(contents))

    uploaded_file = './temp_img/fastapi_post.png'
    pil_image.save(uploaded_file, 'PNG')

    # アップロードした画像をMNIST画像仕様に変換する
    digit_MNIST_img = my_cvtmnist.CvToMNIST_img(uploaded_file)

    # MNIST画像に描かれている数字を予測する
    # prediction = my_predict.PredictNumber(digit_MNIST_img)
    prediction = my_predict.PredictNumber2(digit_MNIST_img)

    print(datetime.now(), "@app.post('/process_image_prediction/')")
    print(f'　　予測結果 on FastAPI       ： {prediction}')
    print(f'　　分類結果 on FastAPI(numpy)： {prediction.numpy()}')
    # print(type(prediction)) # <class 'torch.Tensor'>

    prediction_str = format(f'{prediction}')
    # print(prediction_str)

    # ヒートマップ画像をローカルに保存し保存した画像ファイルを表示する
    # my_predict.eXplainableAI(digit_MNIST_img)
    my_predict.eXplainableAI2(digit_MNIST_img)

    print(datetime.now(), '-- Retrun --:')
    return {'prediction_number': prediction_str}


# MNIST仕様変換した画像を返す
@app.get('/process_image_mnist/')
async def process_image_mnist():
    print(datetime.now(), "@app.get('/process_image_mnist/')")
    # MNIST変換画像を戻す
    with open('./temp_img/thresh28x28.png', 'rb') as f:
        return Response(content=f.read(), media_type='image/png')


# Shapで作ったヒートマップ画像を返す
@app.get('/process_image_xai/')
async def process_image_xai():
    print(datetime.now(), "@app.get('/process_image_xai/')")
    # Shapeヒートマップ画像を戻す
    with open('./temp_img/shap_plot white.png', 'rb') as f:
        return Response(content=f.read(), media_type='image/png')


print(datetime.now(), "This main.py code for FastAPI ends.")
print('/// \n')