import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import os
import csv
import foresnet as resnet


class M3DV(data.Dataset):
    def __init__(self,img_path,label_path,label_name):
        self.img_path = img_path
        self.label_path = label_path + '/' + label_name
        self.label_name = label_name

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
        self.tr_set, self.tr_label = self.load_traindata()
        self.tr_set, self.tr_label = self.data_augmentation(self.tr_set, self.tr_label, 0)
        self.tr_set = np.array(self.tr_set).astype('float32') / 255
        self.tr_label = np.array(self.tr_label)
        print('Data loaded.')
        print('length of train set: {}'.format(len(self.tr_set)))
        print('length of train label: {}'.format(len(self.tr_label)))
        self.length = len(self.tr_set)

    # 这里load的数据没有归一化
    def load_traindata(self):
        size = 32
        x_return = np.zeros((465, size, size, size))
        x_name = []
        labels = []
        with open(self.label_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            row_idx = 0
            for row in reader:
                if row_idx == 0:
                    row_idx += 1
                    continue
                x_name.append(row[0])
                labels.append(float(row[1]))

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
        return x_return, labels

    def data_augmentation(self, x, y, index):
        x_lr = x_ud = x_r1 = x_r2 = x_r3 = x_r4 = x_r5 = x_r6 = x_m1 = x_m2 = x_m3 = x_m4 = np.zeros(np.shape(x))
        y_m1 = y_m2 = y_m3 = y_m4 = np.zeros(np.shape(y), 'float')

        l1 = np.random.beta(0.1, 0.1, len(x))
        l2 = np.random.beta(0.15, 0.15, len(x))
        l3 = np.random.beta(0.2, 0.2, len(x))
        l4 = np.random.beta(0.25, 0.25, len(x))
        randi = np.random.randint(0, len(x), len(x))
        for ii in range(x.shape[0]):
            x_lr[ii, :, :, :] = np.fliplr(x[ii, :, :, :])
            x_ud[ii, :, :, :] = np.flipud(x[ii, :, :, :])
            x_r1[ii, :, :, :] = np.rot90(x[ii, :, :, :], 1, (0, 1))
            x_r2[ii, :, :, :] = np.rot90(x[ii, :, :, :], 1, (0, 2))
            x_r3[ii, :, :, :] = np.rot90(x[ii, :, :, :], 1, (1, 2))
            x_r4[ii, :, :, :] = np.rot90(x[ii, :, :, :], 3, (0, 1))
            x_r5[ii, :, :, :] = np.rot90(x[ii, :, :, :], 3, (0, 2))
            x_r6[ii, :, :, :] = np.rot90(x[ii, :, :, :], 3, (1, 2))
            x_m1[ii] = x[ii] * l1[ii] + (1 - l1[ii]) * x[randi[ii]]
            y_m1[ii] = y[ii] * l1[ii] + (1 - l1[ii]) * y[randi[ii]]
            x_m2[ii] = x[ii] * l2[ii] + (1 - l2[ii]) * x[randi[ii]]
            y_m2[ii] = y[ii] * l2[ii] + (1 - l2[ii]) * y[randi[ii]]
            x_m3[ii] = x[ii] * l3[ii] + (1 - l3[ii]) * x[randi[ii]]
            y_m3[ii] = y[ii] * l3[ii] + (1 - l3[ii]) * y[randi[ii]]
            x_m4[ii] = x[ii] * l4[ii] + (1 - l4[ii]) * x[randi[ii]]
            y_m4[ii] = y[ii] * l4[ii] + (1 - l4[ii]) * y[randi[ii]]
        # 这里只取了两个方向的翻转、两个方向的90度旋转，还有四个mixup
        x_train = np.r_[x, x_lr, x_ud, x_r1, x_r2, x_m1, x_m2, x_m3, x_m4]
        y_train = np.r_[y, y, y, y, y, y_m1, y_m2, y_m3, y_m4]
#        x_train = np.r_[x, x_lr, x_ud, x_r1, x_r2]
#        y_train = np.r_[y, y, y, y, y]
        # 这里的应该是验证集的
        x_test = np.r_[x, x_lr, x_ud, x_r1, x_r2, x_r3, x_r4, x_r5, x_r6]
        y_test = np.r_[y, y, y, y, y, y, y, y, y]
        if index == 0:
            return x_train, y_train
        else:
            return x_test, y_test

    def __getitem__(self, index):
        data = np.array([self.tr_set[index]])
        label = np.array([self.tr_label[index]])
        return torch.from_numpy(data).float(), torch.from_numpy(label).long()

    def __len__(self):
        #返回数据长度
        return self.length


def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()
    average_loss = 0
    batch_idx = 0
    for i, (data,label) in enumerate(train_loader):
        data_cuda = data.cuda()
        label_cuda = label.cuda()
        # compute output
        outputs = model(data_cuda)
        label_cuda = label_cuda.squeeze()
        loss = criterion(outputs, label_cuda)
        average_loss += loss
        batch_idx += 1
        # print('In epoch {}, batch {}, loss is {}'.format(epoch, i, loss))

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('average loss is:{}'.format(average_loss/batch_idx))

def valid(valid_loader, model):
    # switch to eval mode
    model.eval()
    tmp = 0
    number = 0
    for i, (data,label) in enumerate(valid_loader):
        data_cuda = data.cuda()
        label_cuda = label.cuda()
        # compute output
        outputs = model(data_cuda)
        normalized = F.sigmoid(outputs)
        for j in range(len(normalized)):
            candidate = normalized[j]
            number += 1
            if candidate[1] > candidate[0]:
                if label[j][0] == 1:
                    tmp += 1
            else:
                if label[j][0] == 0:
                    tmp += 1
    
    print('Valid:{}'.format(tmp/number))
    return tmp/number

def save_checkpoint(states,  output_dir, filename='checkpoint.pth'):
    torch.save(states, os.path.join(output_dir, filename))

def reload(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)['state_dict']
    new_state_dict = model.state_dict()
    for k in checkpoint.keys():
        new_k = k.replace('module.', '')
        if new_k in new_state_dict.keys():
            # 检测字符串是否以指定字符开头
            new_state_dict[new_k] = checkpoint[k]
            print('Successfully load {}!'.format(new_k))
    model.load_state_dict(new_state_dict)

    return model

def load_resume(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)['state_dict']
    epoch = torch.load(checkpoint_path)['epoch']
    new_state_dict = model.state_dict()
    for k in checkpoint.keys():
        new_k = k.replace('module.', '')
        if new_k in new_state_dict.keys():
            # 检测字符串是否以指定字符开头
            new_state_dict[new_k] = checkpoint[k]
            print('Successfully load {}!'.format(new_k))
    model.load_state_dict(new_state_dict)

    return model, epoch

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, output, target):
        new_output = F.sigmoid(output)
        loss = -target * torch.log10(new_output[:,1]) - (1-target) * torch.log10(1-new_output[:,1])
        loss = torch.mean(loss)

        return loss


if __name__ == "__main__":
    k_means = 1
    data_path = "data/train_val"
    label_path = "data/"
    label_name = "train_val.csv"
    test_path = "test"
    final_output_dir = "models/model10_random"
    pretrain_path = 'models/pretrained/resnet_10.pth'
    resume_path = 'models/model10_random/newest_model.pth'
    pretrain = True
    resume = False
    weight_decay = 0.01

    for k in range(k_means):
        print('K={}'.format(k))
        data_set = M3DV(data_path,label_path,label_name)
        train_size = int(0.8 * len(data_set))
        valid_size = len(data_set) - train_size
        train_dataset, valid_dataset = data.random_split(data_set, [train_size, valid_size])

        num_epoches = 100
        batch_size = 32
        learning_rate = 0.0001

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            # 内存多的设备用True
            pin_memory=True
        )

        valid_loader = data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            # 内存多的设备用True
            pin_memory=True
        )
        
        model = resnet.resnet10(
                    num_classes=2,
                    shortcut_type='B',
                    sample_size=100,
                    sample_duration=100)

        start_epoch = 0
        if pretrain:
            model = reload(model, pretrain_path)
        
        if resume:
            model, start_epoch = load_resume(model, resume_path)

        model.cuda()

        # 加入正则化
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = CrossEntropy()

        maxacc = 0
        for epoch in range(start_epoch, num_epoches):
            print('epoch:'+str(epoch+1))
            train(train_loader, model, criterion, optimizer, epoch)
            valid_acc = valid(valid_loader, model)
            save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, final_output_dir, 'newest_model.pth')
            if valid_acc > maxacc:
                print("save model")
                maxacc = valid_acc
                save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, final_output_dir, "model1219-d10-k"+str(k) + '.pth')

