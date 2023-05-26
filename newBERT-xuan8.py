import random
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
import pandas as pd
from torch import nn

import torch
from torch import nn


device = "cuda:1"
torch.manual_seed(0)

df=pd.read_csv('/workspace/SFLmailCorpus.csv',header=0,sep=',') 
# print(df.head())
# data_text = torch.tensor(df['本文'].values.astype(np.float32))
# data_label = torch.tensor(df['Level'].values.astype(np.float32))



data_text = df['メール本文'].tolist()
data_label =[
    df['受信者の社会的立場'].tolist(),
    df['送信者の社会的立場'].tolist(),
    df['送信者身分'].tolist(),
    df['受信者身分'].tolist(),
    df['内外関係'].tolist(),
    df['送信者数'].tolist(),
    df['受信者数'].tolist(),
    df['送信者の動き\t'].tolist(),
    df['送信者の動き(詳細)'].tolist(),
    df['やりとりにおける役割'].tolist(),
    df['やりとりされるもの'].tolist()]



# label
label = [[data_label[l][index] for l in range(len(data_label))] for index in range(len(data_label[0]))]


lable_list =[]
for index in range(len(data_label[0])):
    if label[:][index] not in lable_list:
        lable_list.append(label[index])


label_to_index = {str(v):k for k,v in enumerate(lable_list)}
index_to_label = {k:v for k,v in enumerate(lable_list)}

data_labelIndex = [label_to_index[str(l)] for l in label]



########################################################

# 日本語の事前学習モデル
MODEL_NAME = 'cl-tohoku/bert-large-japanese'
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
# bert_sc = BertForSequenceClassification.from_pretrained(
#     MODEL_NAME, num_labels=500
# )
# bert_sc = bert_sc.to(device)

# bert_sc = nn.Sequential(net ,nn.Linear(500, 150))



class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_0 = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=150).to(device)
        # self.out = nn.Linear(500, 150).to(device)


    def forward(self, **X):
        X = self.bert_0(**X)
        return X.logits

bert_sc = MyNet()



# tokenizer
max_length = 128*2
dataset_for_loader = []
for i in range(len(data_text)):
    encoding = tokenizer(
        data_text[i], 
        max_length=max_length, 
        padding='max_length',
        truncation=True
        #return_tensors='pt'
    ) 
    encoding['labels'] = data_labelIndex[i] # ラベルを追加
    # encoding = { k: torch.tensor(v) for k, v in encoding.items() }
    encoding = { k: torch.tensor(v).to(device) for k, v in encoding.items() }
    dataset_for_loader.append(encoding)


# データセットの分割
random.shuffle(dataset_for_loader) # ランダムにシャッフル
n = len(dataset_for_loader)
n_train = int(0.6*n)
n_val = int(0.2*n)
dataset_train = dataset_for_loader[:n_train] # 学習データ
dataset_val = dataset_for_loader[n_train:n_train+n_val] # 検証データ
dataset_test = dataset_for_loader[n_train+n_val:] # テストデータ


# 学習データはshuffle=Trueにする。
dataloader_train = DataLoader(
    dataset_train, batch_size=16, shuffle=True
) 
dataloader_val = DataLoader(dataset_val, batch_size=1)
dataloader_test = DataLoader(dataset_test, batch_size=32)

# for idx, batch in enumerate(dataloader_train):
#     print(f'# batch {idx}')
#     print(batch)
#     break


lr = 1e-5
num_epochs = 20
trainer = torch.optim.Adam(bert_sc.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')

for epoch in range(num_epochs):
    # train
    bert_sc.train()
    for idx, batch in enumerate(dataloader_train):
        trainer.zero_grad()
        # print(f"epoch:{epoch}; batch:{idx}")
        prediction = bert_sc(**batch)
        l = loss(prediction, batch["labels"]).mean()
        # l = prediction.loss
        l.sum().backward()
        trainer.step()
        print(f"epoch:{epoch}; batch:{idx}; loss:{l}")  
    #     break
    # break



    # eval
    bert_sc.eval()
    acc_count, num_count = 0, 0
    prediction_labels_all = torch.tensor([]).to(device)
    labels_all = torch.tensor([]).to(device)
    for idx, batch in enumerate(dataloader_val):
        prediction = bert_sc(**batch)
        prediction_labels = torch.tensor([prediction[i].argmax() for i in range(len(prediction))]).to(device)
        # acc_count += (batch["labels"] == prediction_labels).sum()
        # num_count += len(batch["labels"])

        prediction_labels_all = torch.cat((prediction_labels_all,prediction_labels), dim=0)
        labels_all = torch.cat((labels_all, batch["labels"]), dim=0)
    
    result_correction = (prediction_labels_all == labels_all).tolist() 
    acc = result_correction.count(True) / len(result_correction)
    print(f"eval: epoch:{epoch}; acc_count:{result_correction.count(True)}; num_epochs:{num_epochs} \n acc:{acc}")


# analyze
result_prediction = [index_to_label[prediction_labels_all[i].item()] for i in range(len(prediction_labels_all))]
labels = [index_to_label[labels_all[i].item()] for i in range(len(labels_all))]


# 写成了2个pandas用来分析结果
prediction = pd.DataFrame(result_prediction, columns=df.columns[2:])
labels = pd.DataFrame(labels, columns=df.columns[2:]) #真正正确的

prediction==labels # 保持原表格大小不变，一样的部分是true

prediction[(prediction==labels).all(axis=1) == True] # 完全一致的部分抽出




%config InlineBackend.figure_format = 'svg'
#混同行列(Confusion Matrix)
for i in set(prediction["受信者の社会的立場"]):
    a = prediction["受信者の社会的立場"] == i
    c = labels["受信者の社会的立場"] [a]
    print(f"{i}\n" ,  c.value_counts())


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import japanize_matplotlib

# 假设您有真实标签和预测标签的数组
y_true = np.array(labels)
y_pred = np.array(prediction)

for i in range(len(y_true[0])):
    set(y_true[:, i])
    tmp = confusion_matrix(y_true[:, i], y_pred[:, i] , labels=list(set(y_true[:, i]))) 

    # 使用seaborn绘制混淆矩阵的热图
    sns.heatmap(tmp, annot=True, fmt='d', xticklabels=list(set(y_true[:, i])), yticklabels=list(set(y_true[:, i])))
    plt.xlabel("Predicted label")
    plt.ylabel("Ture label")
    # 显示图像
    plt.show()

#真のラベル配列と予測ラベル配列の各列について、マクロF1スコアを評価
from sklearn.metrics import f1_score
for i in range(y_true.shape[1]):
    f1 = f1_score(y_true[:, i], y_pred[:, i], average='macro')
    print("F1 Score for column {}: {:.4f}".format(i, f1))


for i in range(y_true.shape[1]):
    accuracy = np.sum(y_true[:, i] == y_pred[:, i]) / len(y_true[:, i])
    print("Accuracy for column {}: {:.4f}".format(i, accuracy))


