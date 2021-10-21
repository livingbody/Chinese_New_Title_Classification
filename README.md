# 常规赛：中文新闻文本标题分类Baseline(PaddleNLP)

# 一.方案介绍

* github地址： [https://github.com/livingbody/Chinese_New_Title_Classification](https://github.com/livingbody/Chinese_New_Title_Classification)
* aistudio地址：[https://aistudio.baidu.com/aistudio/projectdetail/2364933?contributionType=1](https://aistudio.baidu.com/aistudio/projectdetail/2364933?contributionType=1)

## 1.1 赛题简介：

文本分类是借助计算机对文本集(或其他实体或物件)按照一定的分类体系或标准进行自动分类标记。本次比赛为新闻标题文本分类 ，选手需要根据提供的新闻标题文本和类别标签训练一个新闻分类模型，然后对测试集的新闻标题文本进行分类，评价指标上使用Accuracy = 分类正确数量 / 需要分类总数量。同时本次参赛选手需使用飞桨框架和飞桨文本领域核心开发库PaddleNLP，PaddleNLP具备简洁易用的文本领域全流程API、多场景的应用示例、非常丰富的预训练模型，深度适配飞桨框架2.x版本。

比赛传送门：[常规赛：中文新闻文本标题分类](https://aistudio.baidu.com/aistudio/competition/detail/107)

## 1.2 数据介绍：

THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。本次比赛数据集在原始新浪新闻分类体系的基础上，重新整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。提供训练数据共832471条。

比赛提供数据集的格式：训练集和验证集格式：原文标题+\t+标签，测试集格式：原文标题。

## 1.3 Baseline思路：

赛题为一道较常规的短文本多分类任务，本项目主要基于PaddleNLP通过预训练模型Robert在提供的训练数据上进行微调完成新闻14分类模型的训练与优化，最后利用训练好的模型对测试数据进行预测并生成提交结果文件。

注意本项目运行需要选择至尊版的GPU环境！若显存不足注意适当改小下batchsize！

BERT前置知识补充：[【原理】经典的预训练模型-BERT](https://aistudio.baidu.com/aistudio/projectdetail/2297740)

![](https://ai-studio-static-online.cdn.bcebos.com/adafc232c53f49258d410e68e3863f1de6747547a9d34ca6b6bf8f4891f4621b)

![](https://ai-studio-static-online.cdn.bcebos.com/b3c5b8ca36e84dcda5a9f1b35b1c6123caf703a1d5b34e1db841bcfdeac2d0d5)

# 二、数据读取与分析

## 1.数据分析


```python
# 进入比赛数据集存放目录
%cd /home/aistudio/data/data103654/
```

    /home/aistudio/data/data103654



```python
# 使用pandas读取数据集
import pandas as pd
train = pd.read_table('train.txt', sep='\t',header=None)  # 训练集
dev = pd.read_table('dev.txt', sep='\t',header=None)      # 验证集
test = pd.read_table('test.txt', sep='\t',header=None)    # 测试集
```


```python
print(f"train数据集长度： {len(train)}\t dev数据集长度{len(dev)}\t test数据集长度{len(test)}")
```

    train数据集长度： 752471	 dev数据集长度80000	 test数据集长度83599



```python
# 添加列名便于对数据进行更好处理
train.columns = ["text_a",'label']
dev.columns = ["text_a",'label']
test.columns = ["text_a"]
```


```python
# 拼接训练和验证集，便于统计分析
total = pd.concat([train,dev],axis=0)
```


```python
#创建字体目录fonts
%cd ~
# !mkdir .fonts

# 复制字体文件到该路径
!cp data/data61659/simhei.ttf .fonts/
```

    /home/aistudio
    cp: cannot create regular file '.fonts/': Not a directory



```python
# 总类别标签分布统计
print(total['label'].value_counts())
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

#指定默认字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.family']='sans-serif'
#解决负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False

total['label'].value_counts().plot.bar()
plt.show()
```

    科技    162245
    股票    153949
    体育    130982
    娱乐     92228
    时政     62867
    社会     50541
    教育     41680
    财经     36963
    家居     32363
    游戏     24283
    房产     19922
    时尚     13335
    彩票      7598
    星座      3515
    Name: label, dtype: int64


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/font_manager.py:1331: UserWarning: findfont: Font family ['sans-serif'] not found. Falling back to DejaVu Sans
      (prop.get_family(), self.defaultFamily[fontext]))



![png](output_12_2.png)



```python
# 最大文本长度
max(total['text_a'].str.len())
```




    48




```python
# 文本长度统计分析,通过分析可以看出文本较短，最长为48
total['text_a'].map(len).describe()
```




    count    832471.000000
    mean         19.388112
    std           4.097139
    min           2.000000
    25%          17.000000
    50%          20.000000
    75%          23.000000
    max          48.000000
    Name: text_a, dtype: float64




```python
# 对测试集的长度统计分析，可以看出在长度上分布与训练数据相近
test['text_a'].map(len).describe()
```




    count    83599.000000
    mean        19.815022
    std          3.883845
    min          3.000000
    25%         17.000000
    50%         20.000000
    75%         23.000000
    max         84.000000
    Name: text_a, dtype: float64



## 2.保存数据


```python
# 保存处理后的数据集文件
train.to_csv('train.csv', sep='\t', index=False)  # 保存训练集，格式为text_a,label
dev.to_csv('dev.csv', sep='\t', index=False)      # 保存验证集，格式为text_a,label
test.to_csv('test.csv', sep='\t', index=False)    # 保存测试集，格式为text_a
```

# 三.基于PaddleNLP构建基线模型

![](https://ai-studio-static-online.cdn.bcebos.com/75742955d912447c948bb679994a09039af5f85cfb554715be56c970db5ec3f6)

## 1 PaddleNLP环境准备


```python
# 下载最新版本的paddlenlp
# !pip install --upgrade paddlenlp
```


```python
# 导入所需的第三方库
import math
import numpy as np
import os
import collections
from functools import partial
import random
import time
import inspect
import importlib
from tqdm import tqdm
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import IterableDataset
from paddle.utils.download import get_path_from_url
# 导入paddlenlp所需的相关包
import paddlenlp as ppnlp
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import MapDataset
from paddle.dataset.common import md5file
from paddlenlp.datasets import DatasetBuilder
```

## 2.数据集定义


```python
# 定义要进行分类的14个类别
label_list=list(train.label.unique())
print(label_list)
```

    ['科技', '体育', '时政', '股票', '娱乐', '教育', '家居', '财经', '房产', '社会', '游戏', '彩票', '星座', '时尚']



```python
# id 到 label转换
id_label_dict={}
for i in range(0,len(label_list)):
    id_label_dict[i]=label_list[i]
print(id_label_dict)
#################################################
# label到id转换
label_id_dict={}
for i in range(0,len(label_list)):
    label_id_dict[label_list[i]]=i
print(label_id_dict)
```

    {0: '科技', 1: '体育', 2: '时政', 3: '股票', 4: '娱乐', 5: '教育', 6: '家居', 7: '财经', 8: '房产', 9: '社会', 10: '游戏', 11: '彩票', 12: '星座', 13: '时尚'}
    {'科技': 0, '体育': 1, '时政': 2, '股票': 3, '娱乐': 4, '教育': 5, '家居': 6, '财经': 7, '房产': 8, '社会': 9, '游戏': 10, '彩票': 11, '星座': 12, '时尚': 13}



```python
def read(pd_data):
    for index, item in pd_data.iterrows():       
        yield {'text_a': item['text_a'], 'label': label_id_dict[item['label']]}
```


```python
# 训练集、测试集
from paddle.io import Dataset, Subset
from paddlenlp.datasets import MapDataset
from paddlenlp.datasets import load_dataset

train_dataset = load_dataset(read, pd_data=train,lazy=False)
dev_dataset = load_dataset(read, pd_data=dev,lazy=False)
```


```python
for i in range(5):
    print(train_dataset[i])
```

    {'text_a': '网易第三季度业绩低于分析师预期', 'label': 0}
    {'text_a': '巴萨1年前地狱重现这次却是天堂 再赴魔鬼客场必翻盘', 'label': 1}
    {'text_a': '美国称支持向朝鲜提供紧急人道主义援助', 'label': 2}
    {'text_a': '增资交银康联 交行夺参股险商首单', 'label': 3}
    {'text_a': '午盘：原材料板块领涨大盘', 'label': 3}


## 3加载预训练模型


```python
# 此次使用在中文领域效果较优的roberta-wwm-ext-large模型，预训练模型一般“大力出奇迹”，选用大的预训练模型可以取得比base模型更优的效果
MODEL_NAME = "roberta-wwm-ext-large"
# 只需指定想要使用的模型名称和文本分类的类别数即可完成Fine-tune网络定义，通过在预训练模型后拼接上一个全连接网络（Full Connected）进行分类
model = ppnlp.transformers.RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=14) # 此次分类任务为14分类任务，故num_classes设置为14
# 定义模型对应的tokenizer，tokenizer可以把原始输入文本转化成模型model可接受的输入数据格式。需注意tokenizer类要与选择的模型相对应，具体可以查看PaddleNLP相关文档
tokenizer = ppnlp.transformers.RobertaTokenizer.from_pretrained(MODEL_NAME)
```

PaddleNLP不仅支持RoBERTa预训练模型，还支持ERNIE、BERT、Electra等预训练模型。具体可以查看：[PaddleNLP模型](https://paddlenlp.readthedocs.io/zh/latest/source/paddlenlp.transformers.html)

下表汇总了目前PaddleNLP支持的各类预训练模型。用户可以使用PaddleNLP提供的模型，完成问答、序列分类、token分类等任务。同时还提供了22种预训练的参数权重供用户使用，其中包含了11种中文语言模型的预训练权重。

| Model | Tokenizer| Supported Task| Model Name|
|---|---|---|---|
| [BERT](https://arxiv.org/abs/1810.04805) | BertTokenizer|BertModel<br> BertForQuestionAnswering<br> BertForSequenceClassification<br>BertForTokenClassification| `bert-base-uncased`<br> `bert-large-uncased` <br>`bert-base-multilingual-uncased` <br>`bert-base-cased`<br> `bert-base-chinese`<br> `bert-base-multilingual-cased`<br> `bert-large-cased`<br> `bert-wwm-chinese`<br> `bert-wwm-ext-chinese` |
|[ERNIE](https://arxiv.org/abs/1904.09223)|ErnieTokenizer<br>ErnieTinyTokenizer|ErnieModel<br> ErnieForQuestionAnswering<br> ErnieForSequenceClassification<br> ErnieForTokenClassification| `ernie-1.0`<br> `ernie-tiny`<br> `ernie-2.0-en`<br> `ernie-2.0-large-en`|
|[RoBERTa](https://arxiv.org/abs/1907.11692)|RobertaTokenizer| RobertaModel<br>RobertaForQuestionAnswering<br>RobertaForSequenceClassification<br>RobertaForTokenClassification| `roberta-wwm-ext`<br> `roberta-wwm-ext-large`<br> `rbt3`<br> `rbtl3`|
|[ELECTRA](https://arxiv.org/abs/2003.10555) |ElectraTokenizer| ElectraModel<br>ElectraForSequenceClassification<br>ElectraForTokenClassification<br>|`electra-small`<br> `electra-base`<br> `electra-large`<br> `chinese-electra-small`<br> `chinese-electra-base`<br>|

注：其中中文的预训练模型有 `bert-base-chinese, bert-wwm-chinese, bert-wwm-ext-chinese, ernie-1.0, ernie-tiny, roberta-wwm-ext, roberta-wwm-ext-large, rbt3, rbtl3, chinese-electra-base, chinese-electra-small` 等。

![](https://ai-studio-static-online.cdn.bcebos.com/0e7623086e034228a1f7e9db834536486ba42d00823749f3bb083f3e25cdf7d9)

## 4.定义数据处理函数


```python
# 定义数据加载和处理函数
def convert_example(example, tokenizer, max_seq_length=128, is_test=False):
    qtconcat = example["text_a"]
    encoded_inputs = tokenizer(text=qtconcat, max_seq_len=max_seq_length)  # tokenizer处理为模型可接受的格式 
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids

# 定义数据加载函数dataloader
def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    # 训练数据集随机打乱，测试数据集不打乱
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
```

# 四、模型训练

## 1.超参设置


```python
# 参数设置：
# 批处理大小，显存如若不足的话可以适当改小该值  
batch_size = 360
# 文本序列最大截断长度，需要根据文本具体长度进行确定，最长不超过512。 通过文本长度分析可以看出文本长度最大为48，故此处设置为48
max_seq_length = max(total['text_a'].str.len())
```

## 2.数据处理


```python
# 将数据处理成模型可读入的数据格式
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length)

batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack()  # labels
): [data for data in fn(samples)]

# 训练集迭代器
train_data_loader = create_dataloader(
    train_dataset,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)

# 验证集迭代器
dev_data_loader = create_dataloader(
    dev_dataset,
    mode='dev',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)
```

## 3.设置评价指标

适用于BERT这类Transformer模型的学习率为warmup的动态学习率。


```python
# 定义超参，loss，优化器等
from paddlenlp.transformers import LinearDecayWithWarmup


# 定义训练过程中的最大学习率
learning_rate = 4e-5
# 训练轮次
epochs = 32
# 学习率预热比例
warmup_proportion = 0.1
# 权重衰减系数，类似模型正则项策略，避免模型过拟合
weight_decay = 0.01

num_training_steps = len(train_data_loader) * epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_proportion)

# AdamW优化器
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=weight_decay,
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ])

criterion = paddle.nn.loss.CrossEntropyLoss()  # 交叉熵损失函数
metric = paddle.metric.Accuracy()              # accuracy评价指标
```

## 4.模型评估

ps：模型训练时，可以通过在终端输入nvidia-smi命令或者通过点击底部‘性能监控’选项查看显存的占用情况，适当调整好batchsize，防止出现显存不足意外暂停的情况。


```python
# 定义模型训练验证评估函数
@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids, token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval loss: %.8f, accu: %.8f" % (np.mean(losses), accu))  # 输出验证集上评估效果
    model.train()
    metric.reset()
    return  np.mean(losses), accu  # 返回准确率
```

## 5.模型训练


```python
# 固定随机种子便于结果的复现
# seed = 1024
seed = 512
random.seed(seed)
np.random.seed(seed)
paddle.seed(seed)
%cd ~
```

    /home/aistudio


ps:模型训练时可以通过在终端输入nvidia-smi命令或通过底部右下的性能监控选项查看显存占用情况，显存不足的话要适当调整好batchsize的值。


```python
# 模型训练：
import paddle.nn.functional as F
from visualdl import LogWriter

save_dir='./'
writer = LogWriter("./log")
tic_train = time.time()
global_step = 0
best_val_acc=0
tic_train = time.time()
accu=0

for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, segment_ids, labels = batch
        logits = model(input_ids, segment_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()
        global_step+=1
        if global_step % 40 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.8f, accu: %.8f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    40 / (time.time() - tic_train)))
            tic_train = time.time()

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()
        
        # 大于100次再eval，或者再大一点eval，太小没有eval的意义
        if global_step % 200 == 0 and global_step>=3000:
            # 评估当前训练的模型
            eval_loss, eval_accu = evaluate(model, criterion, metric, dev_data_loader)
            print("eval  on dev  loss: {:.8}, accu: {:.8}".format(eval_loss, eval_accu))
            # 加入eval日志显示
            writer.add_scalar(tag="eval/loss", step=global_step, value=eval_loss)
            writer.add_scalar(tag="eval/acc", step=global_step, value=eval_accu)
            # 加入train日志显示
            writer.add_scalar(tag="train/loss", step=global_step, value=loss)
            writer.add_scalar(tag="train/acc", step=global_step, value=acc)
            save_dir = "best_checkpoint"
            # 加入保存       
            if eval_accu>best_val_acc:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                best_val_acc=eval_accu
                print(f"模型保存在 {global_step} 步， 最佳eval准确度为{best_val_acc:.8f}！")
                save_param_path = os.path.join(save_dir, 'best_model.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                fh = open('best_checkpoint/best_model.txt', 'w', encoding='utf-8')
                fh.write(f"模型保存在 {global_step} 步， 最佳eval准确度为{best_val_acc:.8f}！")
                fh.close()

tokenizer.save_pretrained(save_dir)
```

    global step 40, epoch: 1, batch: 40, loss: 2.66168261, accu: 0.03826389, speed: 0.61 step/s
    global step 80, epoch: 1, batch: 80, loss: 2.55217147, accu: 0.06201389, speed: 0.60 step/s
    global step 120, epoch: 1, batch: 120, loss: 2.35012722, accu: 0.13446759, speed: 0.60 step/s
    global step 160, epoch: 1, batch: 160, loss: 1.87099862, accu: 0.22362847, speed: 0.60 step/s
    global step 200, epoch: 1, batch: 200, loss: 1.30100632, accu: 0.30144444, speed: 0.59 step/s
    global step 240, epoch: 1, batch: 240, loss: 0.89265573, accu: 0.37986111, speed: 0.60 step/s
    global step 280, epoch: 1, batch: 280, loss: 0.59178758, accu: 0.44602183, speed: 0.59 step/s
    global step 320, epoch: 1, batch: 320, loss: 0.40790802, accu: 0.49969618, speed: 0.60 step/s
    global step 360, epoch: 1, batch: 360, loss: 0.39611995, accu: 0.54423611, speed: 0.59 step/s
    global step 400, epoch: 1, batch: 400, loss: 0.40274370, accu: 0.58047917, speed: 0.59 step/s
    global step 440, epoch: 1, batch: 440, loss: 0.31529644, accu: 0.61082702, speed: 0.59 step/s
    global step 480, epoch: 1, batch: 480, loss: 0.30232269, accu: 0.63619213, speed: 0.58 step/s
    global step 520, epoch: 1, batch: 520, loss: 0.33462012, accu: 0.65837073, speed: 0.60 step/s
    global step 560, epoch: 1, batch: 560, loss: 0.29469413, accu: 0.67740079, speed: 0.60 step/s
    global step 600, epoch: 1, batch: 600, loss: 0.26175639, accu: 0.69389815, speed: 0.60 step/s
    global step 640, epoch: 1, batch: 640, loss: 0.23396534, accu: 0.70868490, speed: 0.59 step/s
    global step 680, epoch: 1, batch: 680, loss: 0.23595652, accu: 0.72179330, speed: 0.59 step/s
    global step 720, epoch: 1, batch: 720, loss: 0.20935695, accu: 0.73346065, speed: 0.59 step/s
    global step 760, epoch: 1, batch: 760, loss: 0.19594315, accu: 0.74396199, speed: 0.60 step/s
    global step 800, epoch: 1, batch: 800, loss: 0.22720760, accu: 0.75360417, speed: 0.60 step/s
    global step 840, epoch: 1, batch: 840, loss: 0.26421335, accu: 0.76210648, speed: 0.59 step/s
    global step 880, epoch: 1, batch: 880, loss: 0.19780147, accu: 0.77011048, speed: 0.60 step/s
    global step 920, epoch: 1, batch: 920, loss: 0.11989931, accu: 0.77750302, speed: 0.59 step/s
    global step 960, epoch: 1, batch: 960, loss: 0.25420505, accu: 0.78407986, speed: 0.59 step/s
    global step 1000, epoch: 1, batch: 1000, loss: 0.21302503, accu: 0.79027500, speed: 0.59 step/s
    global step 1040, epoch: 1, batch: 1040, loss: 0.21804300, accu: 0.79607639, speed: 0.59 step/s
    global step 1080, epoch: 1, batch: 1080, loss: 0.20004293, accu: 0.80146348, speed: 0.60 step/s
    global step 1120, epoch: 1, batch: 1120, loss: 0.13202177, accu: 0.80650298, speed: 0.59 step/s
    global step 1160, epoch: 1, batch: 1160, loss: 0.20018557, accu: 0.81110872, speed: 0.59 step/s
    global step 1200, epoch: 1, batch: 1200, loss: 0.17520264, accu: 0.81549306, speed: 0.60 step/s
    global step 1240, epoch: 1, batch: 1240, loss: 0.16838190, accu: 0.81971550, speed: 0.59 step/s
    global step 1280, epoch: 1, batch: 1280, loss: 0.19400227, accu: 0.82351128, speed: 0.59 step/s
    global step 1320, epoch: 1, batch: 1320, loss: 0.12447893, accu: 0.82720328, speed: 0.59 step/s
    global step 1360, epoch: 1, batch: 1360, loss: 0.24491167, accu: 0.83060049, speed: 0.58 step/s
    global step 1400, epoch: 1, batch: 1400, loss: 0.16221967, accu: 0.83382937, speed: 0.60 step/s
    global step 1440, epoch: 1, batch: 1440, loss: 0.22842427, accu: 0.83685571, speed: 0.60 step/s
    global step 1480, epoch: 1, batch: 1480, loss: 0.16180661, accu: 0.83975976, speed: 0.59 step/s
    global step 1520, epoch: 1, batch: 1520, loss: 0.20264953, accu: 0.84254751, speed: 0.59 step/s
    global step 1560, epoch: 1, batch: 1560, loss: 0.19902995, accu: 0.84518519, speed: 0.60 step/s
    global step 1600, epoch: 1, batch: 1600, loss: 0.17133461, accu: 0.84770660, speed: 0.60 step/s
    global step 1640, epoch: 1, batch: 1640, loss: 0.14073460, accu: 0.85010501, speed: 0.59 step/s
    global step 1680, epoch: 1, batch: 1680, loss: 0.16109636, accu: 0.85238922, speed: 0.59 step/s
    global step 1720, epoch: 1, batch: 1720, loss: 0.17589858, accu: 0.85458979, speed: 0.59 step/s
    global step 1760, epoch: 1, batch: 1760, loss: 0.17276089, accu: 0.85674558, speed: 0.59 step/s
    global step 1800, epoch: 1, batch: 1800, loss: 0.15035772, accu: 0.85872068, speed: 0.59 step/s
    global step 1840, epoch: 1, batch: 1840, loss: 0.15502724, accu: 0.86067633, speed: 0.59 step/s
    global step 1880, epoch: 1, batch: 1880, loss: 0.14147128, accu: 0.86254433, speed: 0.59 step/s
    global step 1920, epoch: 1, batch: 1920, loss: 0.16157188, accu: 0.86439525, speed: 0.59 step/s
    global step 1960, epoch: 1, batch: 1960, loss: 0.16061194, accu: 0.86607993, speed: 0.60 step/s
    global step 2000, epoch: 1, batch: 2000, loss: 0.16157156, accu: 0.86773472, speed: 0.59 step/s
    global step 2040, epoch: 1, batch: 2040, loss: 0.20375380, accu: 0.86926743, speed: 0.60 step/s
    global step 2080, epoch: 1, batch: 2080, loss: 0.13566890, accu: 0.87075187, speed: 0.59 step/s
    global step 2120, epoch: 2, batch: 29, loss: 0.13396244, accu: 0.87233111, speed: 0.60 step/s
    global step 2160, epoch: 2, batch: 69, loss: 0.11650663, accu: 0.87395393, speed: 0.60 step/s
    global step 2200, epoch: 2, batch: 109, loss: 0.15937009, accu: 0.87548613, speed: 0.59 step/s
    global step 2240, epoch: 2, batch: 149, loss: 0.16712877, accu: 0.87698592, speed: 0.59 step/s
    global step 2280, epoch: 2, batch: 189, loss: 0.10917982, accu: 0.87835751, speed: 0.59 step/s
    global step 2320, epoch: 2, batch: 229, loss: 0.09748576, accu: 0.87969137, speed: 0.59 step/s
    global step 2360, epoch: 2, batch: 269, loss: 0.15784366, accu: 0.88097528, speed: 0.59 step/s
    global step 2400, epoch: 2, batch: 309, loss: 0.14650443, accu: 0.88217818, speed: 0.59 step/s
    global step 2440, epoch: 2, batch: 349, loss: 0.18103148, accu: 0.88336440, speed: 0.59 step/s
    global step 2480, epoch: 2, batch: 389, loss: 0.13677049, accu: 0.88451571, speed: 0.59 step/s
    global step 2520, epoch: 2, batch: 429, loss: 0.14471813, accu: 0.88569441, speed: 0.59 step/s
    global step 2560, epoch: 2, batch: 469, loss: 0.10740256, accu: 0.88682323, speed: 0.59 step/s
    global step 2600, epoch: 2, batch: 509, loss: 0.11426187, accu: 0.88785854, speed: 0.59 step/s
    global step 2640, epoch: 2, batch: 549, loss: 0.12956497, accu: 0.88887509, speed: 0.60 step/s
    global step 2680, epoch: 2, batch: 589, loss: 0.15131475, accu: 0.88988513, speed: 0.60 step/s
    global step 2720, epoch: 2, batch: 629, loss: 0.08550296, accu: 0.89083788, speed: 0.59 step/s
    global step 2760, epoch: 2, batch: 669, loss: 0.13498443, accu: 0.89177710, speed: 0.59 step/s
    global step 2800, epoch: 2, batch: 709, loss: 0.10268699, accu: 0.89267359, speed: 0.59 step/s
    global step 2840, epoch: 2, batch: 749, loss: 0.14280342, accu: 0.89357809, speed: 0.60 step/s
    global step 2880, epoch: 2, batch: 789, loss: 0.14486991, accu: 0.89449220, speed: 0.60 step/s
    global step 2920, epoch: 2, batch: 829, loss: 0.14648315, accu: 0.89538125, speed: 0.59 step/s
    global step 2960, epoch: 2, batch: 869, loss: 0.10042919, accu: 0.89624438, speed: 0.59 step/s
    global step 3000, epoch: 2, batch: 909, loss: 0.17139210, accu: 0.89707801, speed: 0.59 step/s
    eval loss: 0.10926867, accu: 0.96518750
    eval  on dev  loss: 0.10926867, accu: 0.9651875
    模型保存在 3000 步， 最佳eval准确度为0.96518750！
    global step 3040, epoch: 2, batch: 949, loss: 0.11478367, accu: 0.95430556, speed: 0.20 step/s
    global step 3080, epoch: 2, batch: 989, loss: 0.13277237, accu: 0.95520833, speed: 0.59 step/s
    global step 3120, epoch: 2, batch: 1029, loss: 0.17795959, accu: 0.95530093, speed: 0.59 step/s
    global step 3160, epoch: 2, batch: 1069, loss: 0.13172887, accu: 0.95578125, speed: 0.59 step/s
    global step 3200, epoch: 2, batch: 1109, loss: 0.12943189, accu: 0.95620833, speed: 0.59 step/s
    eval loss: 0.10214896, accu: 0.96726250
    eval  on dev  loss: 0.10214896, accu: 0.9672625
    模型保存在 3200 步， 最佳eval准确度为0.96726250！
    global step 3240, epoch: 2, batch: 1149, loss: 0.15281750, accu: 0.95611111, speed: 0.20 step/s
    global step 3280, epoch: 2, batch: 1189, loss: 0.16772594, accu: 0.95597222, speed: 0.59 step/s
    global step 3320, epoch: 2, batch: 1229, loss: 0.09810641, accu: 0.95634259, speed: 0.59 step/s
    global step 3360, epoch: 2, batch: 1269, loss: 0.15749517, accu: 0.95569444, speed: 0.60 step/s
    global step 3400, epoch: 2, batch: 1309, loss: 0.09842104, accu: 0.95615278, speed: 0.59 step/s
    eval loss: 0.09931073, accu: 0.96733750
    eval  on dev  loss: 0.099310733, accu: 0.9673375
    模型保存在 3400 步， 最佳eval准确度为0.96733750！
    global step 3440, epoch: 2, batch: 1349, loss: 0.10282571, accu: 0.95791667, speed: 0.20 step/s
    global step 3480, epoch: 2, batch: 1389, loss: 0.12941405, accu: 0.95770833, speed: 0.60 step/s
    global step 3520, epoch: 2, batch: 1429, loss: 0.19096760, accu: 0.95699074, speed: 0.59 step/s
    global step 3560, epoch: 2, batch: 1469, loss: 0.16284712, accu: 0.95699653, speed: 0.60 step/s
    global step 3600, epoch: 2, batch: 1509, loss: 0.16376908, accu: 0.95715278, speed: 0.59 step/s
    eval loss: 0.09319762, accu: 0.96950000
    eval  on dev  loss: 0.093197621, accu: 0.9695
    模型保存在 3600 步， 最佳eval准确度为0.96950000！
    global step 3640, epoch: 2, batch: 1549, loss: 0.14038211, accu: 0.95715278, speed: 0.20 step/s
    global step 3680, epoch: 2, batch: 1589, loss: 0.15250742, accu: 0.95614583, speed: 0.60 step/s
    global step 3720, epoch: 2, batch: 1629, loss: 0.14195631, accu: 0.95606481, speed: 0.59 step/s
    global step 3760, epoch: 2, batch: 1669, loss: 0.20458704, accu: 0.95640625, speed: 0.59 step/s
    global step 3800, epoch: 2, batch: 1709, loss: 0.12256803, accu: 0.95740278, speed: 0.60 step/s
    eval loss: 0.09174430, accu: 0.96953750
    eval  on dev  loss: 0.091744296, accu: 0.9695375
    模型保存在 3800 步， 最佳eval准确度为0.96953750！
    global step 3840, epoch: 2, batch: 1749, loss: 0.11225484, accu: 0.95770833, speed: 0.20 step/s
    global step 3880, epoch: 2, batch: 1789, loss: 0.09859297, accu: 0.95878472, speed: 0.60 step/s
    global step 3920, epoch: 2, batch: 1829, loss: 0.14603040, accu: 0.95803241, speed: 0.60 step/s
    global step 3960, epoch: 2, batch: 1869, loss: 0.10136303, accu: 0.95729167, speed: 0.60 step/s
    global step 4000, epoch: 2, batch: 1909, loss: 0.11340237, accu: 0.95754167, speed: 0.59 step/s
    eval loss: 0.09867395, accu: 0.96693750
    eval  on dev  loss: 0.098673955, accu: 0.9669375
    global step 4040, epoch: 2, batch: 1949, loss: 0.12067087, accu: 0.95604167, speed: 0.20 step/s
    global step 4080, epoch: 2, batch: 1989, loss: 0.13655968, accu: 0.95774306, speed: 0.59 step/s
    global step 4120, epoch: 2, batch: 2029, loss: 0.11788758, accu: 0.95921296, speed: 0.59 step/s
    global step 4160, epoch: 2, batch: 2069, loss: 0.10230584, accu: 0.95947917, speed: 0.60 step/s
    global step 4200, epoch: 3, batch: 18, loss: 0.10166528, accu: 0.95939256, speed: 0.59 step/s
    eval loss: 0.08537734, accu: 0.97185000
    eval  on dev  loss: 0.085377336, accu: 0.97185
    模型保存在 4200 步， 最佳eval准确度为0.97185000！
    global step 4240, epoch: 3, batch: 58, loss: 0.13083500, accu: 0.96687500, speed: 0.19 step/s
    global step 4280, epoch: 3, batch: 98, loss: 0.09505958, accu: 0.96732639, speed: 0.59 step/s
    global step 4320, epoch: 3, batch: 138, loss: 0.08153034, accu: 0.96763889, speed: 0.60 step/s
    global step 4360, epoch: 3, batch: 178, loss: 0.09627575, accu: 0.96762153, speed: 0.59 step/s
    global step 4400, epoch: 3, batch: 218, loss: 0.10256484, accu: 0.96801389, speed: 0.59 step/s
    eval loss: 0.08025850, accu: 0.97300000
    eval  on dev  loss: 0.080258496, accu: 0.973
    模型保存在 4400 步， 最佳eval准确度为0.97300000！
    global step 4440, epoch: 3, batch: 258, loss: 0.05031985, accu: 0.96847222, speed: 0.20 step/s
    global step 4480, epoch: 3, batch: 298, loss: 0.10324056, accu: 0.96888889, speed: 0.59 step/s
    global step 4520, epoch: 3, batch: 338, loss: 0.11842163, accu: 0.96819444, speed: 0.59 step/s
    global step 4560, epoch: 3, batch: 378, loss: 0.08254157, accu: 0.96826389, speed: 0.59 step/s
    global step 4600, epoch: 3, batch: 418, loss: 0.08742398, accu: 0.96823611, speed: 0.59 step/s
    eval loss: 0.08798651, accu: 0.97100000
    eval  on dev  loss: 0.087986514, accu: 0.971
    global step 4640, epoch: 3, batch: 458, loss: 0.08975404, accu: 0.96618056, speed: 0.20 step/s
    global step 4680, epoch: 3, batch: 498, loss: 0.07753605, accu: 0.96638889, speed: 0.59 step/s
    global step 4720, epoch: 3, batch: 538, loss: 0.09331383, accu: 0.96664352, speed: 0.60 step/s
    global step 4760, epoch: 3, batch: 578, loss: 0.12421456, accu: 0.96576389, speed: 0.60 step/s
    global step 4800, epoch: 3, batch: 618, loss: 0.09366854, accu: 0.96629167, speed: 0.59 step/s
    eval loss: 0.08481101, accu: 0.97181250
    eval  on dev  loss: 0.084811009, accu: 0.9718125
    global step 4840, epoch: 3, batch: 658, loss: 0.07656927, accu: 0.96430556, speed: 0.20 step/s
    global step 4880, epoch: 3, batch: 698, loss: 0.17020339, accu: 0.96409722, speed: 0.59 step/s
    global step 4920, epoch: 3, batch: 738, loss: 0.08426637, accu: 0.96532407, speed: 0.59 step/s
    global step 4960, epoch: 3, batch: 778, loss: 0.12548208, accu: 0.96505208, speed: 0.60 step/s
    global step 5000, epoch: 3, batch: 818, loss: 0.11333205, accu: 0.96559722, speed: 0.59 step/s
    eval loss: 0.07665095, accu: 0.97478750
    eval  on dev  loss: 0.076650955, accu: 0.9747875
    模型保存在 5000 步， 最佳eval准确度为0.97478750！
    global step 5040, epoch: 3, batch: 858, loss: 0.08936185, accu: 0.96437500, speed: 0.20 step/s
    global step 5080, epoch: 3, batch: 898, loss: 0.16134185, accu: 0.96357639, speed: 0.59 step/s
    global step 5120, epoch: 3, batch: 938, loss: 0.13989289, accu: 0.96402778, speed: 0.60 step/s
    global step 5160, epoch: 3, batch: 978, loss: 0.16866900, accu: 0.96449653, speed: 0.60 step/s
    global step 5200, epoch: 3, batch: 1018, loss: 0.11867096, accu: 0.96465278, speed: 0.59 step/s
    eval loss: 0.07556443, accu: 0.97507500
    eval  on dev  loss: 0.075564429, accu: 0.975075
    模型保存在 5200 步， 最佳eval准确度为0.97507500！
    global step 5240, epoch: 3, batch: 1058, loss: 0.09558195, accu: 0.96659722, speed: 0.20 step/s
    global step 5280, epoch: 3, batch: 1098, loss: 0.12015187, accu: 0.96472222, speed: 0.60 step/s
    global step 5320, epoch: 3, batch: 1138, loss: 0.13201657, accu: 0.96490741, speed: 0.59 step/s
    global step 5360, epoch: 3, batch: 1178, loss: 0.08133844, accu: 0.96442708, speed: 0.60 step/s
    global step 5400, epoch: 3, batch: 1218, loss: 0.10419378, accu: 0.96409722, speed: 0.60 step/s
    eval loss: 0.07599997, accu: 0.97441250
    eval  on dev  loss: 0.075999968, accu: 0.9744125
    global step 5440, epoch: 3, batch: 1258, loss: 0.13161643, accu: 0.96604167, speed: 0.20 step/s
    global step 5480, epoch: 3, batch: 1298, loss: 0.09048895, accu: 0.96569444, speed: 0.59 step/s
    global step 5520, epoch: 3, batch: 1338, loss: 0.11265116, accu: 0.96511574, speed: 0.60 step/s
    global step 5560, epoch: 3, batch: 1378, loss: 0.11892670, accu: 0.96569444, speed: 0.59 step/s
    global step 5600, epoch: 3, batch: 1418, loss: 0.13979697, accu: 0.96529167, speed: 0.59 step/s
    eval loss: 0.07110831, accu: 0.97777500
    eval  on dev  loss: 0.071108311, accu: 0.977775
    模型保存在 5600 步， 最佳eval准确度为0.97777500！
    global step 5640, epoch: 3, batch: 1458, loss: 0.10632215, accu: 0.96381944, speed: 0.20 step/s
    global step 5680, epoch: 3, batch: 1498, loss: 0.07234745, accu: 0.96468750, speed: 0.60 step/s
    global step 5720, epoch: 3, batch: 1538, loss: 0.14574963, accu: 0.96407407, speed: 0.59 step/s
    global step 5760, epoch: 3, batch: 1578, loss: 0.15499011, accu: 0.96421875, speed: 0.59 step/s
    global step 5800, epoch: 3, batch: 1618, loss: 0.12859030, accu: 0.96405556, speed: 0.60 step/s
    eval loss: 0.07231185, accu: 0.97627500
    eval  on dev  loss: 0.072311848, accu: 0.976275
    global step 5840, epoch: 3, batch: 1658, loss: 0.07545812, accu: 0.96611111, speed: 0.20 step/s
    global step 5880, epoch: 3, batch: 1698, loss: 0.13331755, accu: 0.96350694, speed: 0.59 step/s
    global step 5920, epoch: 3, batch: 1738, loss: 0.14839043, accu: 0.96342593, speed: 0.59 step/s
    global step 5960, epoch: 3, batch: 1778, loss: 0.05173485, accu: 0.96317708, speed: 0.60 step/s
    global step 6000, epoch: 3, batch: 1818, loss: 0.10502592, accu: 0.96300000, speed: 0.60 step/s
    eval loss: 0.07389978, accu: 0.97516250
    eval  on dev  loss: 0.073899783, accu: 0.9751625



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-40-461b823aef57> in <module>
         27             tic_train = time.time()
         28 
    ---> 29         loss.backward()
         30         optimizer.step()
         31         lr_scheduler.step()


    <decorator-gen-246> in backward(self, grad_tensor, retain_graph)


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/wrapped_decorator.py in __impl__(func, *args, **kwargs)
         23     def __impl__(func, *args, **kwargs):
         24         wrapped_func = decorator_func(func)
    ---> 25         return wrapped_func(*args, **kwargs)
         26 
         27     return __impl__


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/framework.py in __impl__(*args, **kwargs)
        225         assert in_dygraph_mode(
        226         ), "We only support '%s()' in dynamic graph mode, please call 'paddle.disable_static()' to enter dynamic graph mode." % func.__name__
    --> 227         return func(*args, **kwargs)
        228 
        229     return __impl__


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/varbase_patch_methods.py in backward(self, grad_tensor, retain_graph)
        237             else:
        238                 core.dygraph_run_backward([self], [grad_tensor], retain_graph,
    --> 239                                           framework._dygraph_tracer())
        240         else:
        241             raise ValueError(


    KeyboardInterrupt: 



```python
save_dir='./'
tokenizer.save_pretrained(save_dir)
```


```python
# 测试最优模型参数在验证集上的分数
evaluate(model, criterion, metric, dev_data_loader)
```

# 五、预测
**此处要重启或者`killall -9 python`来释放缓存**

## 1.导入各种类库


```python
# 导入所需的第三方库
import math
import numpy as np
import os
import collections
from functools import partial
import random
import time
import inspect
import importlib
from tqdm import tqdm
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import IterableDataset
from paddle.utils.download import get_path_from_url
# 导入paddlenlp所需的相关包
import paddlenlp as ppnlp
from paddlenlp.data import JiebaTokenizer, Pad, Stack, Tuple, Vocab
from paddlenlp.datasets import MapDataset
from paddle.dataset.common import md5file
from paddlenlp.datasets import DatasetBuilder
```

## 2.加载模型


```python
# 此次使用在中文领域效果较优的roberta-wwm-ext-large模型，预训练模型一般“大力出奇迹”，选用大的预训练模型可以取得比base模型更优的效果
MODEL_NAME = "roberta-wwm-ext-large"
# 只需指定想要使用的模型名称和文本分类的类别数即可完成Fine-tune网络定义，通过在预训练模型后拼接上一个全连接网络（Full Connected）进行分类
model = ppnlp.transformers.RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_classes=14) # 此次分类任务为14分类任务，故num_classes设置为14
# 定义模型对应的tokenizer，tokenizer可以把原始输入文本转化成模型model可接受的输入数据格式。需注意tokenizer类要与选择的模型相对应，具体可以查看PaddleNLP相关文档
tokenizer = ppnlp.transformers.RobertaTokenizer.from_pretrained(MODEL_NAME)
```


```python
# 加载在验证集上效果最优的一轮的模型参数
import os
import paddle

seed = 1024
random.seed(seed)
np.random.seed(seed)
paddle.seed(seed)

params_path = '88.73671/best_checkpoint/best_model.pdparams'
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
```

## 3.加载test数据集


```python
# 读取要进行预测的测试集文件
import pandas as pd
test = pd.read_csv('~/data/data103654/test.txt', header=None, names=['text_a'])  
```


```python
print(max(test['text_a'].str.len()))
```


```python
label_list=['科技', '体育', '时政', '股票', '娱乐', '教育', '家居', '财经', '房产', '社会', '游戏', '彩票', '星座', '时尚']
print(label_list)
```


```python
# 定义要进行分类的类别
id_label_dict={}
for i in range(0,len(label_list)):
    id_label_dict[i]=label_list[i]
print(id_label_dict)
```


```python
!head -n5 ~/data/data103654/test.txt
```

## 5.数据处理


```python
# 定义数据加载和处理函数
def convert_example(example, tokenizer, max_seq_length=48, is_test=False):
    qtconcat = example["text_a"]
    encoded_inputs = tokenizer(text=qtconcat, max_seq_len=max_seq_length)  # tokenizer处理为模型可接受的格式 
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids

# 定义模型预测函数
def predict(model, data, tokenizer, label_map, batch_size=1):
    examples = []
    # 将输入数据（list格式）处理为模型可接受的格式
    for text in data:
        input_ids, segment_ids = convert_example(
            text,
            tokenizer,
            max_seq_length=48,
            is_test=True)
        examples.append((input_ids, segment_ids))

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input id
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # segment id
    ): fn(samples)

    # Seperates data into some batches.
    batches = []
    one_batch = []
    for example in examples:
        one_batch.append(example)
        if len(one_batch) == batch_size:
            batches.append(one_batch)
            one_batch = []
    if one_batch:
        # The last batch whose size is less than the config batch_size setting.
        batches.append(one_batch)

    results = []
    model.eval()
    for batch in batches:
        input_ids, segment_ids = batchify_fn(batch)
        input_ids = paddle.to_tensor(input_ids)
        segment_ids = paddle.to_tensor(segment_ids)
        logits = model(input_ids, segment_ids)
        probs = F.softmax(logits, axis=1)
        idx = paddle.argmax(probs, axis=1).numpy()
        idx = idx.tolist()
        labels = [label_map[i] for i in idx]
        results.extend(labels)
    return results  # 返回预测结果
```


```python
# 定义对数据的预处理函数,处理为模型输入指定list格式
def preprocess_prediction_data(data):
    examples = []
    for text_a in data:
        examples.append({"text_a": text_a})
    return examples

# 对测试集数据进行格式处理
data1 = list(test.text_a)
examples = preprocess_prediction_data(data1)
```

## 6.开始预测


```python
# 对测试集进行预测
results = predict(model, examples, tokenizer, id_label_dict, batch_size=128)   
```


```python
print(results)
```


```python
# 将list格式的预测结果存储为txt文件，提交格式要求：每行一个类别
def write_results(labels, file_path):
    with open(file_path, "w", encoding="utf8") as f:
        f.writelines("\n".join(labels))

write_results(results, "./result.txt")
```


```python
# 因格式要求为zip，故需要将结果文件压缩为submission.zip提交文件
!zip 'submission.zip' 'result.txt'
```


```python
!head result.txt
```

需注意此次要求提交格式为zip，在主目录下找到生成的**submission.zip**文件下载到本地并到比赛页面进行提交即可！

# 四.提升方向：

1.可以针对训练数据进行数据增强从而增大训练数据量以提升模型泛化能力。[NLP Chinese Data Augmentation 一键中文数据增强工具](https://github.com/425776024/nlpcda/)

2.可以在基线模型的基础上通过调参及模型优化进一步提升效果。[文本分类上分微调技巧实战](https://mp.weixin.qq.com/s/CDIeTz6ETpdQEzjdeBW93g)

3.可以尝试使用不同的预训练模型如ERNIE和NEZHA等，并对多模型的结果进行投票融合。[竞赛上分Trick-结果融合](https://aistudio.baidu.com/aistudio/projectdetail/2315563)

4.可以将训练和验证集进行拼接后自定义训练和验证集的划分构建差异性结果用于融合或尝试5folds交叉验证等。

5.取多模型预测结果中相同的部分作为伪标签用于模型的训练。伪标签技巧一般用于模型精度较高时，初学者慎用。

6.有能力的可以尝试在训练语料下重新预训练以及修改模型网络结构等进一步提升效果。

7.更多的技巧可以通过学习其他类似短文本分类比赛的Top分享，多做尝试。[零基础入门NLP - 新闻文本分类](https://tianchi.aliyun.com/competition/entrance/531810/forum)

关于PaddleNLP的使用：建议多看官方最新文档 [PaddleNLP文档](https://paddlenlp.readthedocs.io/zh/latest/get_started/quick_start.html)

PaddleNLP的github地址：[https://github.com/PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) 有问题的话可以在github上提issue，会有专人回答。

# 五、心得

*  1.在初始阶段没必要eval，所以可设条件在1000batch以后再eval
*  2.batch size不宜过大，数据量大，在后续epoch时显存占用会波动上去

