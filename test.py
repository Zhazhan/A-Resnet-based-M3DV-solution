import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import os
import math
import csv
import argparse
import foresnet as resnet


def parse_args():
    parser = argparse.ArgumentParser(description='load config')
    parser.add_argument('--datapath',
                        help='Dataset dir',
                        required=True,
                        type=str)
    parser.add_argument('--modeldir',
                        help='path to stored model',
                        required=True,
                        type=str)

    args = parser.parse_args()
    return args


class M3DV(data.Dataset):
    def __init__(self,img_path, multi):
        self.img_path = img_path
        self.label_path = label_path
        self.label_name = label_name
        self.multi = multi

        #图像尺寸
        self.img_rows = 32
        self.img_cols = 32
        self.img_dims = 32
        
        self.length=0
        self.name = []
        self.img=[]
        self.label=[]
        self.mask=[]
        self.data_get()

    def data_get(self):
        self.tr_set, self.names = self.load_testdata()
        if self.multi:
            self.tr_set, self.names = self.multi_test(self.tr_set, self.names)

        self.tr_set = np.array(self.tr_set).astype('float32') / 255
        print('Data loaded.')
        self.length = len(self.tr_set)

    def multi_test(self, data, name):
        new_data = []
        new_name = []
        for i in range(len(data)):
            new_data.append(data[i])
            new_name.append(name[i])
            aug_data = np.fliplr(data[i, :, :, :])
            new_data.append(aug_data)
            new_name.append(name[i])
            aug_data = np.flipud(data[i, :, :, :])
            new_data.append(aug_data)
            new_name.append(name[i])
            aug_data = np.rot90(data[i, :, :, :], 1, (0, 1))
            new_data.append(aug_data)
            new_name.append(name[i])
            aug_data = np.rot90(data[i, :, :, :], 1, (0, 2))
            new_data.append(aug_data)
            new_name.append(name[i])
        
        return new_data, new_name

    # 这里load的数据没有归一化
    def load_testdata(self):
        size = 32
        x_return = np.zeros((117, size, size, size))
        x_name = []
        with open("sampleSubmission.csv", 'r') as csvfile:
            reader = csv.reader(csvfile)
            row_idx = 0
            for row in reader:
                if row_idx == 0:
                    row_idx += 1
                    continue
                x_name.append(row[0])

        count_file = 0
        # 对训练集中的每个文件
        for i in range(len(x_name)):
            x_file_temp = os.path.join(self.img_path, x_name[i]+'.npz')
            x_voxel = np.array(np.load(x_file_temp)['voxel'])
            x_mask = np.array(np.load(x_file_temp)['seg'])
            # 处理过的输入图像
            x_temp = x_voxel*x_mask*0.8+x_voxel*0.2

            list_xx = x_mask * x_voxel
            # 这里是只包含结节处索引的数组
            list_xx_nz = np.array(np.nonzero(list_xx))
            coor1min = list_xx_nz[0, :].min()
            coor1max = list_xx_nz[0, :].max()
            # x方向结节的长度
            coor1len = coor1max - coor1min + 1
            coor1bigger = coor1len - size
            # 如果结节长度大于32 裁剪一下 缩成32
            if coor1bigger > 0:
                coor1min += coor1bigger // 2
                coor1max -= coor1bigger - coor1bigger // 2
                coor1len = size
            coor1low = (size // 2) - (coor1len // 2)
            coor1high = coor1low + coor1len
            coor2min = list_xx_nz[1, :].min()
            coor2max = list_xx_nz[1, :].max()
            # y方向结节的长度
            coor2len = coor2max - coor2min + 1
            coor2bigger = coor2len - size
            if coor2bigger > 0:
                coor2min += coor2bigger // 2
                coor2max -= coor2bigger - coor2bigger // 2
                coor2len = size
            coor2low = (size // 2) - (coor2len // 2)
            coor2high = coor2low + coor2len
            coor3min = list_xx_nz[2, :].min()
            coor3max = list_xx_nz[2, :].max()
            coor3len = coor3max - coor3min + 1
            coor3bigger = coor3len - size
            if coor3bigger > 0:
                coor1min += coor3bigger // 2
                coor3max -= coor3bigger - coor3bigger // 2
                coor3len = size
            coor3low = (size // 2) - (coor3len // 2)
            coor3high = coor3low + coor3len
            coorlist1 = 0
            for coor1 in range(coor1low, coor1high):
                coorlist2 = 0
                for coor2 in range(coor2low, coor2high):
                    coorlist3 = 0
                    for coor3 in range(coor3low, coor3high):
                        # 将结节移到中心
                        x_return[count_file, coor1, coor2, coor3] = x_temp[
                            coor1min + coorlist1, coor2min + coorlist2, coor3min + coorlist3]
                        coorlist3 += 1
                    coorlist2 += 1
                coorlist1 += 1
            count_file += 1
        return x_return, x_name

    def __getitem__(self, index):
        data = np.array([self.tr_set[index]])
        name = self.names[index]
        print(name)

        return torch.from_numpy(data).float(), name


    def __len__(self):
        #返回数据长度
        return self.length


def reload(model, checkpoint):
    new_state_dict = model.state_dict()
    for k in checkpoint.keys():
        if k in new_state_dict.keys():
            # 检测字符串是否以指定字符开头
            new_state_dict[k] = checkpoint[k]
            print('Successfully load {}!'.format(k))
    model.load_state_dict(new_state_dict)

    return model


def write_csv(data, csv_path):
    new_data = []
    candidates = list(data.keys())
    index = []
    for i in range(len(candidates)):
        index.append(int(candidates[i][9:]))
    idx = np.argsort(index)
    for i in range(len(candidates)):
        name = candidates[idx[i]]
        new_data.append([name, data[name]])

    with open(csv_path, 'w', newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['name', 'predicted'])
        for i in range(len(data)):
            writer.writerow(new_data[i])


def combine_results(all_predicts):
    new_predict = {}
    # 对每个目标
    for i in all_predicts[0].keys():
        score = 0
        # 对每个模型的预测输出
        for j in range(len(all_predicts)):
            score += all_predicts[j][i]
        new_predict[i] = score / len(all_predicts)

    return new_predict


if __name__ == "__main__":
    args = parse_args()
    data_path = args.datapath + '/' + 'train_val'
    label_path = args.datapath
    label_name = "sampleSubmission.csv"
    test_path = args.datapath + '/' + 'test'
    final_output_dir = args.modeldir
    result_path = 'submission.csv'
    multi_test = False
    batch_size = 1
    k = 1

    test_data_set = M3DV(test_path, multi_test)
    test_loader = data.DataLoader(
        test_data_set,batch_size=batch_size,
        shuffle=False,num_workers=0,pin_memory=True
    )
    
    model = resnet.resnet10(
                num_classes=2,
                shortcut_type='B',
                sample_size=100,
                sample_duration=100)

    all_predicts = []
    for k_idx in range(k):
        pretrained_path = args.modeldir + '/model1219-d10-k' + str(k_idx) + '.pth'
        print("start loading model")
        assert os.path.isfile(pretrained_path), "Model does not exist!"
        checkpoint = torch.load(pretrained_path)
        model = reload(model, checkpoint['state_dict'])
        model.cuda().eval()
        
        print("start predicting")

        predicts = {}
        tmp = 0
        for i, (data,name) in enumerate(test_loader):
            data_cuda = data.cuda()
            # compute output
            output = model(data_cuda)
            normalized = F.sigmoid(output)
            candidate = name[0].split('.')[0]
            print(candidate)
            print('Prob is:{}'.format(normalized[0][1]))
            if normalized[0][1] > 0.5:
                tmp += 1

            if multi_test:
                if candidate in predicts.keys():
                    predicts[candidate] += normalized[0][1].item() / 5
                else:
                    predicts[candidate] = normalized[0][1].item() / 5
            else:
                predicts[candidate] = normalized[0][1].item()
        print(tmp)
        all_predicts.append(predicts)

    print('Combine results')
    combined_pre = combine_results(all_predicts)
    write_csv(combined_pre, result_path)

