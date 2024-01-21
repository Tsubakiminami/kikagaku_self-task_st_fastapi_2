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
    VS CODEの設定を開く方法:
    メニューから Code -> 基本設定 -> 設定 を選び設定画面を表示する。
    extra paths で検索する
    Python > Analysis: Extra Paths 下部の「項目の追加」ボタンをクリックする
     /Users/ik/Downloads/virtual_env/venv_Streamlit/lib/python3.10/site-packages/ を入力してOKをクリック、
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

app.py
 このコードでは、Streamlitを使用し撮影した手書き数字画像(リクエストボディ)をサーバへ送信し、数字の予測結果とXAIヒートマップ画像を受け取ります。
 予測結果は文字、MNIST仕様に変換した画像とぁいヒートマップ画像はバイト列で送受信します。
 この三つを一つのレスポンスボディにして受け取りたかったが、上手くできないので個々に受け取ることにした。
 また、画像をバイト列に変換するのに、画像をローカルディスクに保存し'rb'オプションで読み込みする方法を取った。

主な機能：
  1. Streamlitを使用したフロンドエンドを提供する
  2. アップロード画像をMNIST画像変換しFastAPIサーバへ送信する
  3. 予測結果とその根拠となるXAIヒートマップ画像を取得する
  4. 各結果をJSON形式でクライアント側から受信する

使用方法：
  1. このコードでフロントエンドを起動
     streamlit run app.py
  2. ブラウザ上で手書き画像をアップロードするか、サンプル画像をクリックする。
  3. クリックすることで結果を表示する     

起動例：
ik@mini 17_自走期間課題FastAPI % source ../venv_Streamlit/bin/activate
(venv_Streamlit) ik@mini 17_自走期間課題FastAPI % streamlit run app.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.2.39:8501

  For better performance, install the Watchdog module:

  $ xcode-select --install
  $ pip install watchdog
            
ターミナルのログ：
2024-01-21 10:22:31.157451 This app.py code starts now.
  Thie app.py main() runs.
   Webにアップロードしたファイル名は： None
   サンプル画像のファイル名は 　　　： None
   　st.session_state[key]= 　　　　： none
2024-01-21 10:22:31.165782 This app.py code ends.
--- 
'''

# 現在の時刻を取得
from datetime import datetime   # pip install datetime
# current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
# Stremlit とコード実行順序の関連を追跡し理解するためのPRINT文（ターミナルに出力される）
print(datetime.now(), 'This app.py code starts now.')

# 必要なモジュールをインポートする
import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os

# 真っ先に行う処理
# タイトルを表示
st.title('数字の画像分類へようこそ')
st.sidebar.write('V1.01 R6(2024)/01/21')
    # V0.01 R5(2023)/12/02 
    # V0.02 R5(2023)/12/03 
    # V0.03 R5(2023)/12/06
    # V0.04 R5(2023)/12/07
    # V1.00 R6(2024)/01/02



# 自作モジュールにアクセスできるようにする
import my1_cvtmnist as my_cvtmnist
import my2_cnn as my_cnn
import my3_predictX as my_predict


# 使い方の説明を表示
st.write('数字一つが写った画像をアップロードしてAIが数字を分類します。そして、どこを見て分類したのかをヒートマップで表示するAPIです。')

col1, col2 = st.columns(2)

# アップロードした画像を表示する
def display_img(uploaded_file, col):
    # # バイトデータとしてファイルを読み取るとき
    # bytes_data = upload_file.getvalue()
    # st.write(bytes_data)

    # アップロードされた画像を表示
    with col:
        st.image(uploaded_file, caption='Uploaded Image.', width=200)
        st.write("この画像から数字を予測します。")


def classify_img(uploaded_file):
    st.write('Step 3')

    # アップロードした画像をMNIST画像仕様に変換する
    digit_MNIST_img = my_cvtmnist.CvToMNIST_img(uploaded_file)
    # MNIST画像変換した画像を表示する
    st.image(digit_MNIST_img, caption='MNIST仕様画像')
    # MNIST画像に描かれている数字を予測する
    prediction = my_predict.PredictNumber(digit_MNIST_img)

    st.sidebar.write(f'予測結果： {prediction}')
    st.write(f'分類結果： {prediction.numpy()}')

    # ヒートマップ画像をローカルに保存し保存した画像ファイルを表示する
    my_predict.eXplainableAI(digit_MNIST_img)
    st.write('上段はMNISTデータセットからランダムに拾った画像で特徴量を例示し、下段がアップロードした画像での特徴量です')
    st.write('青色はマイナスに影響、赤色はプラスに影響を与えている部分らしいです')
    st.image('shap_plot white.png', caption='eXplainableAI', use_column_width=True)


# 2024/01/02 FastAPIサーバーへ画像を送信し、予測結果を受信しる関数を追加 V1.00
def predict_img(upload_file):
    st.write('New Step 3')

    # FastAPIサーバーのベースURL
    fastapi_base_url = 'http://127.0.0.1:8000' + '/'
    # fastapi_base_url = 'https://kikagaku-self-task-st-fastapi.onrender.com' + '/'
    # # FastAPIのエンドポイントを呼び出す
    # response = requests.get(fastapi_base_url)
    # # FastAPIからのレスポンスを表示
    # if response.status_code == 200:    
    #     result = response.json()
    #     # "message" キーが存在するか確認して表示
    #     if 'message' in result:
    #         st.write("FastAPIからのメッセージ:")
    #         st.text(result["message"])
    #     else:
    #         st.write("FastAPIからのメッセージが見つかりませんでした。")
    # else:
    #     st.write(f"エラー: {response.status_code} - {response.text}")            
    
    f = open(upload_file, 'rb')
    url =  fastapi_base_url + 'process_image_prediction/'
    file = {"file": f}

    # 予測する数字画像をpost
    response = requests.post(url, files=file)
    # FastAPIからのレスポンスを確認
    if response.status_code == 200:
        result = response.json()
        if 'prediction_number' in result:
            st.sidebar.write(f'予測結果： {result["prediction_number"]}')
            st.write(f'分類結果： {result["prediction_number"]}')
        else:
            st.write('返事はありません')
    else:
        st.write(f"エラー: {response.status_code} - {response.text}")


    # MNIST変換画像をget
    url = fastapi_base_url + 'process_image_mnist/'
    response = requests.get(url)
    # FastAPIからのレスポンスを確認
    if response.status_code == 200:
        # 取得した画像を表示
        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes)
        st.image(image, caption="MNIST変換画像")
    else:
        st.write(f"エラー: {response.status_code} - {response.text}")

    # Shapヒートマップ画像をget
    url = fastapi_base_url + 'process_image_xai/'
    response = requests.get(url)
    # FastAPIからのレスポンスを確認
    if response.status_code == 200:
        # 取得した画像を表示
        image_bytes = BytesIO(response.content)
        image = Image.open(image_bytes)
        st.write('上段がアップロードした画像での特徴量で、下段はMNISTデータセットからランダムに拾った画像で特徴量を例示してます。')
        st.write('青色はマイナスに影響、赤色はプラスに影響を与えている部分らしいです')
        st.image(image, caption='eXplainableAI', use_column_width=True)
    else:
        st.write(f"エラー: {response.status_code} - {response.text}")


def main():
    # 作業フォルダ作成
    dir_temp_img = 'temp_img'
    if not os.path.exists(dir_temp_img):
        os.makedirs(dir_temp_img)

    uploaded_file_select = './temp_img/uploaded.png'

    # 各UIを定義するコードを最初に集める。
    # ファイルを選択するUI
    with col1:
        st.write('Step 1')
        uploaded_file = st.file_uploader('画像ファイルを選択してください。', type=['png','jpg','jpeg'])

        if uploaded_file:
            with open(uploaded_file_select, 'wb') as f:
                f.write(uploaded_file.read())

    # サンプルファイル名を選ぶボタン
    btn_s0 = st.sidebar.button('サンプル画像0', key='btn_s0')
    btn_s1 = st.sidebar.button('サンプル画像1', key='btn_s1')
    btn_s2 = st.sidebar.button('サンプル画像2', key='btn_s2')
    btn_s3 = st.sidebar.button('サンプル画像3', key='btn_s3')
    btn_s4 = st.sidebar.button('サンプル画像4', key='btn_s4')
    btn_s5 = st.sidebar.button('サンプル画像5', key='btn_s5')
    btn_s6 = st.sidebar.button('サンプル画像6', key='btn_s6')
    btn_s7 = st.sidebar.button('サンプル画像7', key='btn_s7')
    btn_s8 = st.sidebar.button('サンプル画像8', key='btn_s8')
    btn_s9 = st.sidebar.button('サンプル画像9', key='btn_s9')
    uploaded_file_sample = None

    # 分類開始するボタン
    st.write('Step 2')
    btn_1 = st.button('分類を進めるには、ここをクリックして下さい。', key='btn_1')


    if btn_s0:
        uploaded_file_sample ='./sample_img/black00.png'
    else:
        if btn_s1:
            uploaded_file_sample ='./sample_img/black01.png'
        else:
            if btn_s2:
                uploaded_file_sample ='./sample_img/black02.png'
            else:
                if btn_s3:
                    uploaded_file_sample ='./sample_img/black03.png'
                else:
                    if btn_s4:
                        uploaded_file_sample ='./sample_img/black04.png'
                    else:
                        if btn_s5:
                            uploaded_file_sample ='./sample_img/black05.png'
                        else:
                            if btn_s6:
                                uploaded_file_sample ='./sample_img/black06.png'
                            else:
                                if btn_s7:
                                    uploaded_file_sample ='./sample_img/black07.png'
                                else:
                                    if btn_s8:
                                        uploaded_file_sample ='./sample_img/black08.png'
                                    else:
                                        if btn_s9:
                                            uploaded_file_sample ='./sample_img/black09.png'
    if uploaded_file_sample is not None:
        st.session_state['key'] = uploaded_file_sample
        print(datetime.now(),'  選択したサンプル画像ファイル名は：',uploaded_file_sample)
        display_img(uploaded_file_sample, col2)       

    # if btn_s1:
    #     uploaded_file_sample = './sample_img/bblack01.png'
    #     if 'key' not in st.session_state:
    #         st.session_state['key'] = uploaded_file_sample

    # 選択画像を表示する
    if uploaded_file is not None:
        if 'key' in st.session_state:
            del st.session_state['key']
        print(datetime.now(),'   アップロードした画像ファイル名は：',uploaded_file)
        display_img(uploaded_file, col2)

    # if btn_s1:
    #     print('uploaded_file_sample ',uploaded_file_sample)
    #     display_img(uploaded_file_sample, col2)

    print('   Webにアップロードしたファイル名は：',uploaded_file)
    print('   サンプル画像のファイル名は 　　　：',uploaded_file_sample)
    if 'key' in st.session_state:
        print('   　st.session_state[key]= 　　　　：', st.session_state['key'])
    else:
        print('   　st.session_state[key]= 　　　　： none')


    # 分類する
    if btn_1:
        if 'key' in st.session_state:
            uploaded_file_sample = st.session_state['key']
            print('     Key分類ファイル名は　　　　　　：', uploaded_file_sample)
            display_img(uploaded_file_sample, col2)   

        if uploaded_file is not None:
            # classify_img(uploaded_file)
            predict_img(uploaded_file_select)
        else:
            if uploaded_file_sample is not None:
                # classify_img(uploaded_file_sample)
                predict_img(uploaded_file_sample)
                # del st.session_state['key']
            else:
                st.write('画像が未選択です。')  

       
        # if classify_file is not None:
        #     print('変数のType',type(classify_file))
        #     classify_img(classify_file)
        # else:
        #     print('画像ファイルが未選択です。')



if __name__ == '__main__':
    print('  Thie app.py main() runs.')
    main()


print(datetime.now(), 'This app.py code ends.')
print('--- \n')
