import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# 定义规则字典
data_dict = {
    "40001": 1.15,
    "40501": 3.45,
    "40502": 5.75,
    "41501": 6.945375,
    "41502": 7.036125,
    "41503": 7.1196,
    "41504": 7.1958,
    "41505": 7.333075,
    "41506": 7.531425,
    "41507": 7.762825,
    "41508": 8.027275,
    "41509": 8.258675,
    "41510": 8.457025,
    "41511": 8.604775,
    "41512": 8.701925,
    "42001": 8.80574575,
    "42002": 8.91623725,
    "43001": 13.971483,
    "23001": 0.06985085,
    "23002": 0.20955255,
    "23003": 0.34925425,
    "23004": 0.48895595,
    "23005": 0.62865765,
    "23006": 0.76835935,
    "23007": 0.90806105,
    "23008": 1.04776275,
    "23009": 1.18746445,
    "23010": 1.32716615,
    "23011": 1.46686785,
    "23012": 1.60656955,
    "23101": 0.06985085,
    "23102": 0.20955255,
    "23103": 0.34925425,
    "23104": 0.48895595,
    "23105": 0.62865765,
    "23106": 0.76835935,
    "23107": 0.90806105,
    "23108": 1.04776275,
    "23109": 1.18746445,
    "23110": 1.32716615,
    "23111": 1.46686785,
    "23112": 1.60656955,
    "30001": 0.4381515,
    "30501": 1.225471,
    "31001": 2.324704,
    "34401": 0.46431155,
    "34201": 1.21437655,
    "34001": 1.849298,
    "33501": 2.5731185,
    "10001": 0.79439,
    "10501": 2.1153396,
    "10701": 2.78240995,
    "15001": 0.2482555,
    "17501": 0.7757995,
    "17502": 1.36171,
    "18001": 2.244277
}
# 计算的初始值

#TODO 添加并扩展字典
def extend(data_dict):
    base = 1.58878
    delta_l = 1.0531192 / 40
    data_dict['10501'] = base + delta_l / 2
    last = data_dict['10501']
    for i in range(2, 41):
        key = f'105{i:02}'  # 生成键名，确保是两位数
        data_dict[key] = last + base + delta_l
        last = data_dict[key]
    return data_dict

def extend_340(data_dict):
    base = 0.9286231 + 0.5715069
    delta_l = 0.698336 / 40
    data_dict['34001'] = base + delta_l / 2
    last = data_dict['34001']
    for i in range(2, 41):
        key = f'340{i:02}'  # 生成键名，确保是两位数
        data_dict[key] = last + base + delta_l
        last = data_dict[key]
    return data_dict

def extend_305(data_dict):
    base = 0.876303
    delta_l = 0.698336 / 40
    data_dict['30501'] = base + delta_l / 2
    last = data_dict['30501']
    for i in range(2, 41):
        key = f'305{i:02}'  # 生成键名，确保是两位数
        data_dict[key] = last + base + delta_l
        last = data_dict[key]
    return data_dict

def extend_344(data_dict):
    base = 0
    delta_l = 0.9286231 / 40
    data_dict['34401'] = base + delta_l / 2
    last = data_dict['34401']

    for i in range(2, 41):
        key = f'344{i:02}'  # 生成键名，确保是两位数
        data_dict[key] = last + base + delta_l
        last = data_dict[key]

    return data_dict

def generate_value(sheet_name,data_dict):
    data_dict = extend(data_dict)
    data_dict = extend_305(data_dict)
    data_dict = extend_340(data_dict)
    data_dict = extend_344(data_dict)
    return data_dict[sheet_name]

def reconstuct_data(var_list, file_path, pipeline, index__pipeline:list):
    table = []
    test_table = []
    for x in os.listdir(file_path):
        if x.endswith('.xlsx'):
            file = os.path.join(file_path,x)
            # 一次性读取所有 sheet 数据，返回字典 {sheet_name: DataFrame}
            excel_data = pd.read_excel(file, sheet_name=None, header=1)
            # 提取时间信息，跳过第一张 sheet
            times = excel_data[list(excel_data.keys())[1]]['time'][:]
            # 获取 sheet 名称列表，排除第一个
            # sheet_names = list(excel_data.keys())[1:-1]
            sheet_names = pipeline
            # httemp = pd.read_excel(file, sheet_name='wallt', header=1)
            # 遍历时间索引
            big_filter = []
            for index, sheet_name in enumerate(sheet_names):
                # 遍历每个 sheet 的数据
                if index == 0:
                    data = excel_data[sheet_name].iloc[:]  # 跳过第一行数据
                    q_t = data['q']
                if sheet_name in index__pipeline:
                    for index_t, t in enumerate(times):
                        data = excel_data[sheet_name].iloc[:]  # 跳过第一行数据
                        x_t = generate_value(sheet_name,data_dict)
                        a_t = data.iloc[index_t]['avol']

                        q_t[index_t] += data.iloc[index_t]['q']
                        vars_t = [data.iloc[index_t][var_name]  if var_name not in big_filter else data.iloc[index_t][var_name]for var_name in var_list]  # 提取所需变量

                        test_table.append([t, x_t, a_t, q_t[index_t]] + vars_t)
                # elif index_t % 5 == 0:
                else:
                    for index_t, t in enumerate(times):
                        data = excel_data[sheet_name].iloc[:]  # 跳过第一行数据
                        x_ = generate_value(sheet_name,data_dict)
                        a_ = data.iloc[index_t]['avol']
                        # a_ = 0.03345587677 #344段
                        # a_ = 0.03345587677 #305段
                        q_t[index_t] += data.iloc[index_t]['q']
                        vars_ = [data.iloc[index_t][var_name] if var_name not in big_filter else data.iloc[index_t][
                                                                                                      var_name]
                                  for var_name in var_list]
                        # vars_ = [data.iloc[index_t][var_name] for var_name in var_list]
                        # httemp_ = httemp.iloc[index_t, index + 2]
                        # 提取所需变量
                        # table.append([t, x_, a_, q_] + vars_ + [httemp_])
                        table.append([t, x_, a_, q_t[index_t]] + vars_)

    return table,test_table


class Mydataset(Dataset):
    def __init__(self,feature,label):
        self.feature = torch.tensor(feature)
        self.label = torch.tensor(label)
        self.scaler = MinMaxScaler()
        self.scalerl = MinMaxScaler()

    def __getitem__(self, item):
        return self.feature[item],self.label[item]

    def __len__(self):
        return len(self.label)

    def normal(self,feature,label):
        feature = self.scaler.fit_transform(feature)
        label = self.scalerl.fit_transform(label)
        return feature,label

    def denormal(self,feature,label):
        feature = self.scaler.inverse_transform(feature)
        label = self.scalerl.inverse_transform(label)
        return feature,label

if __name__ == '__main__':
    file_path = r"data\算例5\10501_40"
    var_list = ['p','rhog','rhof','velgj','velfj','voidg',   'fwalgj','fwalfj', 'ug','uf']

    tabel,_ = reconstuct_data(var_list,file_path, pipeline='4', index__pipeline=['42002'])
    feature = np.array(tabel)[:,:4]
    label = np.array(tabel)[:,4:]
    Dataset_ = Mydataset(feature,label)
    All = DataLoader(Dataset_,batch_size=32,shuffle=False,num_workers=0)
