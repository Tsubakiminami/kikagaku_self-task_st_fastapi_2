ここに掲載したコードは、「DXを推進するAI・データサイエンス人材育成コース」自走期間課題の成果物です。
関連URLs：
 - https://www.kikagaku.ai/learning/courses/detail/longterm3/
 - https://www.kyufu.mhlw.go.jp/kensaku/SSR103Scr02M

このコースはハローワークで事前申請しておき、コースの課題をクリアすれば厚生労働省の雇用保険を使って専門実践教育訓練給付金を受けられます。
コース期間は6ヶ月間で前半３ヶ月間はeラーニングと講師による毎週のライブオンライン講座で学び、後半３ヶ月は自走期間と称して成果物を卒業研究みたく作ります。

成果物は２種類から選びます。
 1. 機械学習を搭載したアプリケーションAPI：データの収集と前処理、モデルの要件定義、アプリケーションの開発、APIの作成までの一連の開発を実践する。
 2. データ分析：データの収集と前処理、仮説立案、データの前処理、分析手法の選択、探索的データ分析および確証的データ分析、施策の立案までの一連のデータ分析の流れを実践する。

私は1.を選びました。

自身で設定した課題はXAIです。
具体的には、
 1. MNISTデータセットで手描き数字を画像分類するモデルを構築し、
 2. このモデルを使って自分で手描きした数字を予測した結果を表示する。
 3. 手描き数字は、スマホで撮影しアップロードする
 4. AIは数字のどこを見て予測したかをヒートマップで表示する。（ここがXAIです）
 5. このアプリをデプロイする。

デプロイ URLs
- Remote repositories on Github : https://github.com/Tsubakiminami/kikagaku_self-task_st_fastapi_2
- Backend server by FastAPI on Render : https://kikagaku-self-task-st-fastapi-2.onrender.com
- Frontend by Streamlit : https://kikagaku-self-task-st-fastapi.streamlit.app/　このURLから開きます。

受講コース期間：令和5年(2023)8月〜令和6年1月

まだ卒業していないので、この内容が要件をクリアするかは2024年1月に分かります。

2024/01/03# kikagaku_self-task_st_fastapi
