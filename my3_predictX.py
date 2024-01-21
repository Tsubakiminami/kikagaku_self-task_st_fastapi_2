# キカガク　DXを推進するAI・データサイエンス人材育成コース　給付金要件　自走期間課題の提出
# 設定課題 XAI

# 数字文字の分類を行うモジュール

# 必要なモジュールをインポートする
# import streamlit as st    # Comment out on 2024/01/03
# shapモジュール、datasetの読み込みに時間がかかるので、見える安心感を作るためにプログレスバーを表示
# progress_text = "Operation in progress. Please wait." # Comment out on 2024/01/03
# my_bar = st.progress(0.1, text=progress_text) # Comment out on 2024/01/03

# 必要なモジュールをインポートする
import shap
import numpy as np

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets

import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import japanize_matplotlib

# 自作モジュールにアクセスできるようにする
# import my1_cvtmnist as my_cvtmnist
import my2_cnn as my_cnn
# import my3_predictX as my_predict


# # プログレスバーを適切に伸ばす
# my_bar.progress(0.5, text=progress_text)    # Comment out on 2024/01/03


batch_size = 64 # デプロイした時のメモリ節約で128->64 とした。
num_epochs = 2
device = torch.device('cpu')

# shapモジュール用とテスト画像用のためのデータセットをロード
# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('mnist_data', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor()
#                    ])),
#     batch_size=batch_size, shuffle=True)

# # プログレスバーを適切に伸ばす
# my_bar.progress(0.85, text=progress_text)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./', train=False, download=True, 
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=batch_size, shuffle=True)


# # プログレスバーを一瞬100％にして瞬消す
# my_bar.progress(1.0, text=progress_text)
# my_bar.empty()


# 画像分類で数字を戻す
# def PredictNumber(im):
#     model = my_cnn.Net1().eval()
#     # 重みの読み込み
#     model.load_state_dict(torch.load('./mnist_pl.pt', map_location=torch.device('cpu')))

#     to_tensor = transforms.ToTensor()
#     img_tensor = to_tensor(im)

#     img_tensor = img_tensor.unsqueeze(0)

#     # 予測の実行
#     y = model(img_tensor)

#     # 確率に変換
#     y = F.softmax(y, dim=1)

#     # 予測ラベル
#     y = torch.argmax(y)

#     return y


# 画像の特徴量ヒートマップをローカルに保存する
# def eXplainableAI(im):
#     model = my_cnn.Net2().eval()
#     # 重みの読み込み
#     model.load_state_dict(torch.load('./mnist_nn.pt', map_location=torch.device('cpu')))


#     to_tensor = transforms.ToTensor()
#     img_tensor = to_tensor(im)

#     img_tensor = img_tensor.unsqueeze(0)


#     # since shuffle=True, this is a random sample of test data
#     batch = next(iter(test_loader))
#     # images, _ = batch
#     images, images_nums = batch


#     background = images[:60]


#     test_images = images[60:61]
#     test_images_nums = images_nums[60:62]


#     # 予測の実行
#     y = model(img_tensor)
#     print('Model 予測結果(Uploaded data)', y)

#     # 確率に変換
#     y = F.softmax(y, dim=1)
#     print('　確率：', y)

#     # 予測ラベル
#     y = torch.argmax(y, dim=1)
#     print('　予測ラベル：', y)

#     # 予測の実行
#     y = model(test_images[0])
#     print('Model 予測結果(Test data)', y)

#     # 確率に変換
#     y = F.softmax(y, dim=1)
#     print('　確率：', y)

#     # 予測ラベル
#     y = torch.argmax(y)
#     print('　予測ラベル：', y)
#     print('--')


#     # テンソルを行方向に連結
#     result_tensor = torch.cat((test_images, img_tensor), dim=0)
#     test_images = result_tensor
#     test_images_nums[1] = y


#     e = shap.DeepExplainer(model, background)
#     shap_values = e.shap_values(test_images)

#     shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
#     test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

#     # plot the feature attributions
#     shap.image_plot(shap_numpy, -test_numpy, show=False)
#     # plt.suptitle('shap 手描き画像をテスト')
#     plt.savefig('./temp_img/shap_plot white.png')
#     # plt.show()

#     return 

# PyTorch Lightning
import lightning as L

# 画像分類で数字を戻す
def PredictNumber2(im):
    # モデルロード
    model = my_cnn.Net4.load_from_checkpoint('./mnist_L.ckpt')
    # 予測
    # trainer = L.Trainer()
    # y = trainer.predict(model, im)
    t_im = transforms.ToTensor()(im)
    y = model(t_im.unsqueeze_(0))
    y = F.softmax(y, dim=1)
    y = torch.argmax(y)

    return y

# 画像の特徴量ヒートマップをローカルに保存する
def eXplainableAI2(im):
    # model = my_cnn.Net2().eval()
    # # 重みの読み込み
    # model.load_state_dict(torch.load('./mnist_nn.pt', map_location=torch.device('cpu')))
    # モデルロード
    model = my_cnn.Net4.load_from_checkpoint('./mnist_L.ckpt')

    tensor_im = transforms.ToTensor()(im)
    tensor_im.unsqueeze_(0)
    print('　　　予測画像shap：', tensor_im.shape)  # torch.Size([1, 28, 28])


    # since shuffle=True, this is a random sample of test data
    batch = next(iter(test_loader))
    # images, _ = batch
    images, images_nums = batch


    background = images[:60]
    test_images = images[:2]
    test_images[0] = tensor_im
    print('　　　Background shap：', background.shape, type(background))  # (バッチサイズ、チャネル、高さ、幅) background.shape
    print('　　　test_images shap：', test_images.shape, type(test_images))  # (バッチサイズ、チャネル、高さ、幅) test_shap_images.shape


    # 予測の実行
    y = model(tensor_im)
    print('　　　Model 予測結果(Uploaded data)', y)

    # 確率に変換
    y = F.softmax(y, dim=1)
    print('　　　　確率：', y)

    # 予測ラベル
    y = torch.argmax(y)
    print('　　　　予測ラベル：', y)

    # 予測の実行
    y = model(test_images[1].unsqueeze(0))
    print('　　　Model 予測結果(Test data)', y)

    # 確率に変換
    y = F.softmax(y, dim=1)
    print('　　　　確率：', y)

    # 予測ラベル
    y = torch.argmax(y)
    print('　　　　予測ラベル：', y)
    print('　　　--')


    e = shap.DeepExplainer(model.model, background)
    # shap_values = e.shap_values(test_images, check_additivity=False)
    shap_values = e.shap_values(test_images)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

    # plot the feature attributions
    shap.image_plot(shap_numpy, -test_numpy, show=False)
    # plt.suptitle('shap 手描き画像をテスト')
    plt.savefig('./temp_img/shap_plot white.png')
    # plt.show()

    return 