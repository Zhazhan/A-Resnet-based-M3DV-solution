### 项目信息

这是我的机器学习大作业代码，我的学号为517021910316.

### 代码说明

代码运行环境为pytorch 0.4以上版本，需求库包括：

```
torch
numpy
csv
argparse
```

代码文件中，foresnet.py为网络结构定义，fotrain.py用作训练，test.py用作测试，测试需在命令行输入：

```
python test.py --datapath ./data/dataset --modeldir ./models/model_random
```

其中--datapath用于指定数据集位置，输入数据集所在路径即可，该路径下的文件结构为：

```
test
train_val
sampleSubmission.csv
train_val.csv
```

--modeldir用于指定模型路径，路径下的文件结构为：

```
model1219-d10-k0.pth
model1219-d10-k1.pth
```

注意test.py需要同一目录下存在sampleSubmission.csv文件，用于生成结果文件submission.csv。准备好数据集与模型后运行test.py即可直接输出结果。

若要训练，修改fotrain.py中的数据集路径、预训练模型路径即可。预训练模型使用的是腾讯MedicalNet项目的resnet10.pth模型，下载地址：https://github.com/Tencent/MedicalNet

### 模型下载

由于模型文件较大，放在了百度云上，链接：https://pan.baidu.com/s/1shCfU_mAmP6b4KTUBCR_ug  提取码：pej1。

