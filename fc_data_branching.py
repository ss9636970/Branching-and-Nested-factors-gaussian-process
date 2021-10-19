import Branching_Models_v2 as GP
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import optim
import importlib
from matplotlib import pyplot as plt

torch.set_printoptions(precision=5)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.ticker import LinearLocator, FormatStrFormatter

import functions as fcs



'''branching and nested factors 用到的小function'''
# 轉換 branching data type 的資料形式
def dataTransfer(W, dataType):
    a = dataType['shareFactors']
    b = dataType['branchingFactors']
    c = sum(dataType['nestedFactors'])
    outputs = [W[:, :a], W[:, a:a + b], W[:, a + b:a + b + c]]
    return outputs

'''目標函式 function'''
# 陳新杰學長文章中的 case

# 目標函式 case 1
def branching_obj_case1(inputs, noise=False):
    shareFactors = inputs[0]
    branchingFactors = inputs[1]
    nestedFactors = inputs[2]
    
    shareOutputs = (-1) * torch.cos(7 / 4 * np.pi * shareFactors) / torch.exp((-1.4) / 4 * shareFactors)
    
    branchingOutputs1 = (branchingFactors == 1) + 0.
    branchingOutputs2 = (branchingFactors == 2) + 0.
    branchingOutputs3 = (branchingFactors == 3) + 0.
    
    w1 = nestedFactors[:, 0:1] * branchingOutputs1
    branchingOutputs1 = (w1 + 5 * torch.sin(2 * np.pi * w1)).view(-1, 1)
    
    w2 = nestedFactors[:, 0:1] * branchingOutputs2
    branchingOutputs2 = (w2 + 5 * torch.sin(3 * np.pi * w2)).view(-1, 1)
    
    w3 = nestedFactors[:, 0:1] * branchingOutputs3
    branchingOutputs3 = (w3 + 5 * torch.sin(1 * np.pi * w3)).view(-1, 1)
    
    if noise:
        n = branchingOutputs3.shape[0]
        noiseTerm = torch.normal(0, 0.5, (n, 1))
        return shareOutputs + branchingOutputs1 + branchingOutputs2 + branchingOutputs3 + noiseTerm

    return shareOutputs + branchingOutputs1 + branchingOutputs2 + branchingOutputs3

def branching_obj_case1_2(inputs, noise=False):   # case 1 乘上一個 -1
    shareFactors = inputs[0]
    branchingFactors = inputs[1]
    nestedFactors = inputs[2]
    
    shareOutputs = (-1) * torch.cos(7 / 4 * np.pi * shareFactors) / torch.exp((-1.4) / 4 * shareFactors)
    
    branchingOutputs1 = (branchingFactors == 1) + 0.
    branchingOutputs2 = (branchingFactors == 2) + 0.
    branchingOutputs3 = (branchingFactors == 3) + 0.
    
    w1 = nestedFactors[:, 0:1] * branchingOutputs1
    branchingOutputs1 = (w1 + 5 * torch.sin(2 * np.pi * w1)).view(-1, 1)
    
    w2 = nestedFactors[:, 0:1] * branchingOutputs2
    branchingOutputs2 = (w2 + 5 * torch.sin(3 * np.pi * w2)).view(-1, 1)
    
    w3 = nestedFactors[:, 0:1] * branchingOutputs3
    branchingOutputs3 = (w3 + 5 * torch.sin(1 * np.pi * w3)).view(-1, 1)
    
    if noise:
        n = branchingOutputs3.shape[0]
        noiseTerm = torch.normal(0, 0.25, (n, 1))
        return (shareOutputs + branchingOutputs1 + branchingOutputs2 + branchingOutputs3 + noiseTerm) * (-1)

    return (shareOutputs + branchingOutputs1 + branchingOutputs2 + branchingOutputs3) * (-1)

# 目標函式 case 6
def branching_obj_case6(inputs, noise=False):
    shareFactors = inputs[0]
    branchingFactors = inputs[1]
    nestedFactors = inputs[2]
    
    branchingOutputs1 = (branchingFactors == 1) + 0.
    branchingOutputs2 = (branchingFactors == 2) + 0.
    branchingOutputs3 = (branchingFactors == 3) + 0.
    
    shareOutputs1 = (-1) * (torch.cos(7 / 4 * np.pi * (shareFactors + 1)) * torch.exp((-1.4) / 4 * (shareFactors + 1))) * branchingOutputs1
    shareOutputs2 = (-1) * (torch.cos(7 / 4 * np.pi * (shareFactors - 1)) * torch.exp((-0.2) * (shareFactors - 1))) * branchingOutputs2
    shareOutputs3 = (-1) * (torch.cos(7 / 4 * np.pi * (shareFactors - 1)) * torch.exp((-0.25) * (shareFactors - 1))) * branchingOutputs3

    w1 = nestedFactors[:, 0:1] * branchingOutputs1
    branchingOutputs1 = (w1 + 5 * torch.sin(2 * np.pi * w1)).view(-1, 1)
    
    w2 = nestedFactors[:, 0:1] * branchingOutputs2
    branchingOutputs2 = (w2 + 5 * torch.sin(3 * np.pi * w2)).view(-1, 1)
    
    w3 = nestedFactors[:, 0:1] * branchingOutputs3
    branchingOutputs3 = (w3 + 5 * torch.sin(1 * np.pi * w3)).view(-1, 1)
    
    if noise:
        n = branchingOutputs3.shape[0]
        noiseTerm = torch.normal(0, 0.5, (n, 1))
        return shareOutputs + branchingOutputs1 + branchingOutputs2 + branchingOutputs3 + noiseTerm

    return (shareOutputs1+ shareOutputs2 + shareOutputs3 + branchingOutputs1 + branchingOutputs2 + branchingOutputs3) * (-1)

# object function case 2
def branching_obj_case2(inputs, noise=False):
    shareFactors = inputs[0]
    branchingFactors = inputs[1]
    nestedFactors = inputs[2]
    
    branchingOutputs1 = (branchingFactors == 1) + 0.
    branchingOutputs2 = (branchingFactors == 2) + 0.
    
    shareOutputs1 = (shareFactors - 0.5) ** 2 * branchingOutputs1
    branchingOutputs1 = (nestedFactors + torch.sin(2 * np.pi * nestedFactors)) * branchingOutputs1
    
    shareOutputs2 = ((-1) * (shareFactors - 0.5) ** 2 + 0.5) * branchingOutputs2
    branchingOutputs2 = (nestedFactors + torch.sin(3 * np.pi * nestedFactors)) * branchingOutputs2

    if noise:
        n = branchingOutputs2.shape[0]
        noiseTerm = torch.normal(0, 0.3, (n, 1))
        return shareOutputs1 + shareOutputs2 + branchingOutputs1 + branchingOutputs2 + noiseTerm
    
    return shareOutputs1 + shareOutputs2 + branchingOutputs1 + branchingOutputs2

def branching_obj_case7(inputs, noise=False):
    shareFactors = inputs[0]
    branchingFactors = inputs[1]
    nestedFactors = inputs[2]
    
    branchingOutputs1 = (branchingFactors == 1) + 0.
    branchingOutputs2 = (branchingFactors == 2) + 0.
    
    shareOutputs1 = (shareFactors - 0.5) ** 2 * branchingOutputs1
    branchingOutputs1 = (nestedFactors + torch.sin(2 * np.pi * nestedFactors)) * branchingOutputs1
    
    shareOutputs2 = ((-1) * (shareFactors - 0.5) ** 2 + 0.5) * branchingOutputs2
    branchingOutputs2 = (nestedFactors + torch.sin(3 * np.pi * nestedFactors)) * branchingOutputs2

    if noise:
        n = branchingOutputs2.shape[0]
        noiseTerm = torch.normal(0, 0.3, (n, 1))
        return shareOutputs1 + shareOutputs2 + branchingOutputs1 + branchingOutputs2 + noiseTerm
    
    return shareOutputs1 + shareOutputs2 + branchingOutputs1 + branchingOutputs2

def branching_obj_bowl(inputs, noise=False):
    shareFactors = inputs[0]
    branchingFactors = inputs[1]
    nestedFactors = inputs[2]
    
    branchingOutputs1 = (branchingFactors == 1) + 0.
    branchingOutputs2 = (branchingFactors == 2) + 0.
    branchingOutputs3 = (branchingFactors == 3) + 0.
    
    shareOutputs1 = ((shareFactors - 0.5) ** 2 + branchingOutputs1 * 2) * branchingOutputs1
    shareOutputs2 = (3 * (shareFactors - 0.5) ** 2 + branchingOutputs2 * 1) * branchingOutputs2
    shareOutputs3 = (6 * (shareFactors - 0.5) ** 2) * branchingOutputs3

    branchingOutputs1 = ((nestedFactors - 0.5) ** 2) * branchingOutputs1
    branchingOutputs2 = (3 * (nestedFactors - 0.5) ** 2) * branchingOutputs2
    branchingOutputs3 = (6 * (nestedFactors - 0.5) ** 2) * branchingOutputs3
    
    if noise:
        n = branchingOutputs3.shape[0]
        noiseTerm = torch.normal(0, 0.5, (n, 1))
        return shareOutputs1 + shareOutputs2 + shareOutputs3 + branchingOutputs1 + branchingOutputs2 + branchingOutputs3 + noiseTerm

    return shareOutputs1 + shareOutputs2 + shareOutputs3 + branchingOutputs1 + branchingOutputs2 + branchingOutputs3

def branching_obj_bowl2(inputs, noise=False):
    shareFactors = inputs[0]
    branchingFactors = inputs[1]
    nestedFactors = inputs[2]
    
    branchingOutputs1 = (branchingFactors == 1) + 0.
    branchingOutputs2 = (branchingFactors == 2) + 0.
    branchingOutputs3 = (branchingFactors == 3) + 0.
    
    shareOutputs1 = ((shareFactors - 0.5 - 0.05) ** 2 + branchingOutputs1 * 0.02) * branchingOutputs1
    shareOutputs2 = (3 * (shareFactors - 0.5 + 0.05) ** 2 + branchingOutputs2 * 0.01) * branchingOutputs2
    shareOutputs3 = (6 * (shareFactors - 0.5) ** 2) * branchingOutputs3

    branchingOutputs1 = ((nestedFactors - 0.5 - 0.05) ** 2) * branchingOutputs1
    branchingOutputs2 = (3 * (nestedFactors - 0.5 + 0.05) ** 2) * branchingOutputs2
    branchingOutputs3 = (6 * (nestedFactors - 0.5) ** 2) * branchingOutputs3
    
    if noise:
        n = branchingOutputs3.shape[0]
        noiseTerm = torch.normal(0, 0.5, (n, 1))
        return shareOutputs1 + shareOutputs2 + shareOutputs3 + branchingOutputs1 + branchingOutputs2 + branchingOutputs3 + noiseTerm

    return shareOutputs1 + shareOutputs2 + shareOutputs3 + branchingOutputs1 + branchingOutputs2 + branchingOutputs3

def branching_obj_wave(inputs, noise=False):
    shareFactors = inputs[0]
    branchingFactors = inputs[1]
    nestedFactors = inputs[2]
    
    branchingOutputs1 = (branchingFactors == 1) + 0.
    branchingOutputs2 = (branchingFactors == 2) + 0.
    branchingOutputs3 = (branchingFactors == 3) + 0.
    
    shareOutputs1 = (20 + (shareFactors - 0.5) ** 2 + torch.cos(2 * np.pi * (nestedFactors - 0.5)) + branchingOutputs1 * 2) * branchingOutputs1
    shareOutputs2 = (20 + 2 * ((shareFactors - 0.5) ** 2 + torch.cos(2 * np.pi * (nestedFactors - 0.5))) + branchingOutputs2 * 1) * branchingOutputs2
    shareOutputs3 = (20 + 4 * ((shareFactors - 0.5) ** 2 + torch.cos(2 * np.pi * (nestedFactors - 0.5)))) * branchingOutputs3

    branchingOutputs1 = ((nestedFactors - 0.5) ** 2 + torch.cos(2 * np.pi * (nestedFactors - 0.5)) * 10) * branchingOutputs1
    branchingOutputs2 = 2 * ((nestedFactors - 0.5) ** 2 + torch.cos(2 * np.pi * (nestedFactors - 0.5)) * 10) * branchingOutputs2
    branchingOutputs3 = 4 * ((nestedFactors - 0.5) ** 2 + torch.cos(2 * np.pi * (nestedFactors - 0.5)) * 10) * branchingOutputs3
    
    if noise:
        n = branchingOutputs3.shape[0]
        noiseTerm = torch.normal(0, 0.5, (n, 1))
        return shareOutputs1 + shareOutputs2 + shareOutputs3 + branchingOutputs1 + branchingOutputs2 + branchingOutputs3 + noiseTerm

    return shareOutputs1 + shareOutputs2 + shareOutputs3 + branchingOutputs1 + branchingOutputs2 + branchingOutputs3

'''資料生成 function'''
# 初始化資料 trainX for case 1 用 LHD
def initial_datas_LHD(n):
    z = torch.tensor([1] * n + [2] * n + [3] * n, dtype=torch.long).view(-1, 1)
    s1, s2, s3 = torch.tensor(fcs.data_LHD(2, n)), torch.tensor(fcs.data_LHD(2, n)), torch.tensor(fcs.data_LHD(2, n))
    s = torch.cat([s1, s2, s3], dim=0)
    s = torch.cat([s, z], dim=1)[:, [0, 2, 1]]
    s = s.type(torch.float)
    return s

# 初始化資料 trainX for case 1
def initial_datas():   #
    x = torch.tensor([0.5, 0.7, 0.8, 0.05, 0.9, 0.25, 1., 0.85, 0.55, 0.1, 0.35, 0.4, 0.65, 0., 0.2]).view(-1, 1)
    z = torch.tensor([1] * 5 + [2] * 5 + [3] * 5, dtype=torch.long).view(-1, 1)
    v = torch.tensor([0.55, 0.75, 0.4, 1., 0.] + [0.55, 0.35, 0.75, 0.1, 0.95] + [0.65, 0.35, 0.85, 0.5, 0.]).view(-1, 1)
    trainX = torch.cat([x, z, v], dim=1)
    return trainX

# # 初始化資料 trainX for case ?
# def initial_datas2():   #
#     x = torch.tensor([0.5, 0.7, 0.8, 0.05, 0.9, 0.25, 1., 0.85, 0.55, 0.1]).view(-1, 1)
#     z = torch.tensor([1] * 5 + [2] * 5, dtype=torch.long).view(-1, 1)
#     v = torch.tensor([0.55, 0.75, 0.4, 1., 0.] + [0.55, 0.35, 0.75, 0.1, 0.95]).view(-1, 1)
#     trainX = torch.cat([x, z, v], dim=1)
#     return trainX


'''配合 model 本身的非主要functions'''
class Result_class:
    def __init__(self):
        self.real = [None, None, None]

    # real plot for object function with branching factor zi
    def real_plot(self, model, zi, Zlim, show=False, file=None):
        if self.real[zi-1] is None:
            x = np.arange(0, 1.01, 0.01)
            v = np.arange(0, 1.01, 0.01)
            x, v = np.meshgrid(x, v)
            x, v = torch.tensor(x), torch.tensor(v)
            a, b = x.shape
            z = torch.tensor([zi] * (x.shape[0] * x.shape[1])).view(-1, 1)     ### 改動 branching factor z 的 level
            x2, v2 = x.view(-1, 1), v.view(-1, 1) 
            w = torch.cat([x2, z, v2], dim=1).type(torch.float)
            self.real[zi-1] = model.objF(model.data_transfer(w), noise=False).view(a, b)

        if show:
            figReal = fcs.plot3d(x.numpy(), v.numpy(), self.real[zi-1].numpy(), Zlim=Zlim,
                    xtitle='share factor x',
                    ytitle='nested factor v',
                    ztitle='f(x, v)')
        
        if file is not None:
            figReal.savefig(file)


    # 初始化 plot 資料點，x v為所有可能資料點，pred為預測矩陣，EI為EI矩陣(with branching factor zi)
    def make_data(self, model, zi):
        x = np.arange(0, 1.01, 0.01)
        v = np.arange(0, 1.01, 0.01)
        x, v = np.meshgrid(x, v)
        x, v = torch.tensor(x), torch.tensor(v)
        a, b = x.shape
        z = torch.tensor([zi] * (x.shape[0] * x.shape[1])).view(-1, 1)     ### 改動 branching factor z 的 level
        x, v = x.view(-1, 1), v.view(-1, 1) 
        w = torch.cat([x, z, v], dim=1).type(torch.float)
        
        pred = model.pred(w).view(a, b)
        EI = model.EI(w).view(a, b)
        x, v = x.view(a, b), v.view(a, b)
        return x, v, pred, EI

    # 同時給予MSE結果和圖形(with branching factor zi)
    def get_result_plot(self,  model, zi, ZlimPred, ZlimEI, filesPred=None, filesEI=None, filesContourEI=None):
        x, v, pred, EI = self.make_data(model, zi)
        self.get_result(model, zi,
                        x=x, v=v, pred=pred, EI=EI)
        self.get_plot(model, zi, ZlimPred, ZlimEI, filesPred=None, filesEI=None, filesContourEI=None,
                        x=x, v=v, pred=pred, EI=EI)


    # 給予pred, EI, contour EI圖形(with branching factor zi)
    def get_plot(self, model, zi, ZlimPred, ZlimEI=None, filesPred=None, filesEI=None, filesContourEI=None,
                x=None, v=None, pred=None, EI=None, loadData=None, cmap=cm.coolwarm):
        
        if x is None:
            x, v, pred, EI = self.make_data(model, zi)

        figPred = fcs.plot3d(x.numpy(), v.numpy(), pred.numpy(), Zlim=ZlimPred,
                        xtitle='share factor x',
                        ytitle='nested factor v', 
                        ztitle='prediction',
                        cmap=cmap)

        if ZlimEI is not None:
            figEI = fcs.plot3d(x.numpy(), v.numpy(), EI.numpy(), Zlim=ZlimEI,
                            xtitle='share factor x',
                            ytitle='nested factor v', 
                            ztitle='EI',
                            cmap=cmap)
        else:
            figEI = fcs.plot3d(x.numpy(), v.numpy(), EI.numpy(),
                            xtitle='share factor x',
                            ytitle='nested factor v', 
                            ztitle='EI',
                            cmap=cmap)

        figContourLineEI = fcs.plot_contour_line(x.numpy(), v.numpy(), EI.numpy(),
                                                trainX=model.trainX,
                                                t=zi,
                                            xtitle='share factor x',
                                            ytitle='nested factor v')
        if filesPred is not None:
            figPred.savefig(filesPred)
        if filesEI is not None:
            figEI.savefig(filesEI)
        if filesContourEI is not None:
            figContourLineEI.savefig(filesContourEI)
        if loadData is not None:
            d = {'xy':[x, v], 'prediction':pred, 'EI':EI}
            torch.save(d, loadData)

    # real contour plot with EIpoints and train points(with branching fator zi)
    def get_EIPoints_real_plot(self, zi, file=None, trainX=None, EIpoints=None, xtitle=None, ytitle=None):
        x = np.arange(0, 1.01, 0.01)
        v = np.arange(0, 1.01, 0.01)
        x, v = np.meshgrid(x, v)

        real = self.real[zi-1]
        minIndex = real.argmin().item()
        minPoint = [x.reshape(-1)[minIndex], v.reshape(-1)[minIndex]]

        figContourLine = fcs.plot_contour_line3(x, v, real.numpy(), trainX=trainX, EIpoints=EIpoints, t=zi, xtitle=xtitle, ytitle=ytitle,
                                                minPoint=minPoint, pointsNumber=True)
        if file is not None:
            figContourLine.savefig(file)

    # MSE結果(with branching factor zi)
    def get_result(self, model, zi,
                        x=None, v=None, pred=None, EI=None):        # zi 控制 branching factor
        if x is None:
            x, v, pred, EI = self.make_data(model, zi)
        EIMax = EI.max()
        EIIndex = EI.argmax().item()
        predMin = pred.min()
        predIndex = pred.argmin().item()
        real = self.real[zi-1]
        realMin = real.min()
        realIndex = real.argmin().item()
        print('EI最大點:', EIMax.item(), 'index:', x.view(-1)[EIIndex].item(), v.view(-1)[EIIndex].item())
        print('pred最小點:', predMin.item(), 'index:', x.view(-1)[predIndex].item(), v.view(-1)[predIndex].item())
        print('real最小點:', realMin.item(), 'index:', x.view(-1)[realIndex].item(), v.view(-1)[realIndex].item())
        MSE = (((pred - real) ** 2).sum() / (pred.shape[0] * pred.shape[1])).item()
        print('MSE:', MSE)

    # 整體 MSE 結果
    def get_result_MSE(self, model):
        W = model.every_points()
        pred = model.pred(W)
        real = model.objF(model.data_transfer(W), noise=False)
        return ((pred - real) ** 2).sum() / (pred.shape[0] * pred.shape[1])

# 不同模型對應各個目標函結果
def result_main(obj_case):
    modelDataType = {'shareFactors':1, 'branchingFactors':1, 'branchingFactorsLevels':[3], 'nestedFactors':[1]}
    
    modelTypes = ['Branching_Kriging', 'Branching_GP']
    trainDatas = torch.load('./data/branching_data/datas.pt')

    ResultM = Result_class()
    for modelType in modelTypes:
        modelMSE = []
        for i, trainData in enumerate(trainDatas):
            trainX = trainData[0]
            trainY = trainData[1]
            if modelType == 'Branching_Kriging':
                model = GP.Branching_Kriging(**modelDataType)
                
            elif modelType == 'Branching_GP':
                model = GP.Branching_GP(**modelDataType)
                
            # elif modelType == 'Branching_Ind':
            #     model = GP.Branching_Ind(**modelDataType)
                
            # elif modelType == 'Branching_QQ':
            #     model = GP.Branching_QQ(**modelDataType)
                
            # elif modelType == 'Branching_MOGP':
            #     model = GP.Branching_MOGP(**modelDataType)

            model.get_obj(obj_case)
            model.train_datas(trainX=trainX, trainY=trainY)
            optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
            
            GP.train(model, optimizer, 1600, show=False)

            ResultM.real_plot(model, 1, [-6, 6], show=False), 
            ResultM.real_plot(model, 2, [-6, 6], show=False), 
            ResultM.real_plot(model, 3, [-6, 6], show=False)

            ResultM.get_EIPoints_real_plot(1, trainX=trainX, EIpoints=model.trainX[30:, :],
                                        file='./pic/case1_2/{}/EIinReal_{}_data{}_z{}'.format(modelType, modelType, i+1, 1),
                                        xtitle='.share factor x', ytitle='nested factor v', minPoint=[0.58, 0.26])
            ResultM.get_EIPoints_real_plot(2, trainX=trainX, EIpoints=model.trainX[30:, :],
                                        file='./pic/case1_2/{}/EIinReal_{}_data{}_z{}'.format(modelType, modelType, i+1, 2),
                                        xtitle='share factor x', ytitle='nested factor v', minPoint=[0.58, 0.84])
            ResultM.get_EIPoints_real_plot(3, trainX=trainX, EIpoints=model.trainX[30:, :],
                                        file='./pic/case1_2/{}/EIinReal_{}_data{}_z{}'.format(modelType, modelType, i+1, 3),
                                        xtitle='share factor x', ytitle='nested factor v', minPoint=[0.58, 0.52])

            ResultM.get_plot(model, 1, ZlimPred=[-6, 6], ZlimEI=[-3, 3], 
                            filesPred='./pic/case1_2/{}/pred_{}_data{}_z{}'.format(modelType, modelType, i+1, 1))
            ResultM.get_plot(model, 2, ZlimPred=[-6, 6], ZlimEI=[-3, 3], 
                            filesPred='./pic/case1_2/{}/pred_{}_data{}_z{}'.format(modelType, modelType, i+1, 2))
            ResultM.get_plot(model, 3, ZlimPred=[-6, 6], ZlimEI=[-3, 3], 
                            filesPred='./pic/case1_2/{}/pred_{}_data{}_z{}'.format(modelType, modelType, i+1, 3))

            dMSE = [ResultM.get_result_MSE(model)]
            for points in range(1, 31):
                GP.train_new_points(model, optimizer, 1000, 1)
                mse = ResultM.get_result_MSE(model)
                dMSE.append(mse)
                if points % 10 == 0:
                    ResultM.get_EIPoints_real_plot(1, trainX=trainX, EIpoints=model.trainX[30:, :],
                                                file='./pic/case1_2/{}/EIinReal_{}_data{}_z{}_p{}'.format(modelType, modelType, i+1, 1, points),
                                                xtitle='.share factor x', ytitle='nested factor v', minPoint=[0.58, 0.26])
                    ResultM.get_EIPoints_real_plot(2, trainX=trainX, EIpoints=model.trainX[30:, :],
                                                file='./pic/case1_2/{}/EIinReal_{}_data{}_z{}_p{}'.format(modelType, modelType, i+1, 2, points),
                                                xtitle='share factor x', ytitle='nested factor v', minPoint=[0.58, 0.84])
                    ResultM.get_EIPoints_real_plot(3, trainX=trainX, EIpoints=model.trainX[30:, :],
                                                file='./pic/case1_2/{}/EIinReal_{}_data{}_z{}_p{}'.format(modelType, modelType, i+1, 3, points),
                                                xtitle='share factor x', ytitle='nested factor v', minPoint=[0.58, 0.52])

                    ResultM.get_plot(model, 1, ZlimPred=[-6, 6], ZlimEI=[-3, 3], 
                                    filesPred='./pic/case1_2/{}/pred_{}_data{}_z{}_p{}'.format(modelType, modelType, i+1, 1, points))
                    ResultM.get_plot(model, 2, ZlimPred=[-6, 6], ZlimEI=[-3, 3], 
                                    filesPred='./pic/case1_2/{}/pred_{}_data{}_z{}_p{}'.format(modelType, modelType, i+1, 2, points))
                    ResultM.get_plot(model, 3, ZlimPred=[-6, 6], ZlimEI=[-3, 3], 
                                    filesPred='./pic/case1_2/{}/pred_{}_data{}_z{}_p{}'.format(modelType, modelType, i+1, 3, points))
            modelMSE.append(dMSE)
        modelMSE = torch.tensor(modelMSE)
        torch.save(modelMSE, './data/{}_MSE.pt'.format(modelType))

# 模型重複多次計算找到幾次最小值
def result_main2(iteration, np):
    modelType = ['Branching_Kriging', 'Branching_GP']
    # modelType = ['Branching_GP']
    obj_case = [branching_obj_case1_2, branching_obj_bowl2]

    modelDataType = {'shareFactors':1, 'branchingFactors':1, 'branchingFactorsLevels':[3], 'nestedFactors':[1]}
    dataType = {'shareFactors':1, 'branchingFactors':1, 'branchingFactorsLevels':[3], 'nestedFactors':[1]}
    trainX = initial_datas()

    getPoints = []
    for m in modelType:
        for obj in obj_case:
            trainY = obj(dataTransfer(trainX, dataType))
            getP = []
            for i in range(iteration):
                if m == 'Branching_Kriging':
                    model = GP.Branching_Kriging(**modelDataType)

                elif m == 'Branching_GP':
                    model = GP.Branching_GP(**modelDataType)

                model.get_obj(obj)
                model.train_datas(trainX=trainX, trainY=trainY)

                t = fun_train(model, obj, np=np)        # 調整 np 決定要用 EI 選幾個點
                getP.append(t)
            getPoints.append(getP)
    return getPoints

# 模型重複多次計算找到幾次最小值，針對兩個不同初始點數量的資料
def result_main2_2(iteration, np):
    models = [GP.Branching_Kriging, GP.Branching_GP]
    obj_case = [branching_obj_case1_2, branching_obj_bowl2]
    trainDataPaths = ['./pic/meet1015/data/points15.pt', './pic/meet1015/data/points30.pt']

    modelType = {'shareFactors':1, 'branchingFactors':1, 'branchingFactorsLevels':[3], 'nestedFactors':[1]}
    
    getPoints = []
    getRatio = []
    for m in models:
        get1 = []
        get1_1 = []
        for obj in obj_case:
            get2 = []
            get1_2 = []
            for path in trainDataPaths:
                trainX = torch.load(path)
                trainY = obj(dataTransfer(trainX, modelType))
                get3 = []
                get1_3 = []
                for i in range(iteration):
                    model = m(**modelType)
                    model.get_obj(obj)
                    model.train_datas(trainX=trainX, trainY=trainY)
                    t = fun_train(model, obj, np=np)        # 調整 np 決定要用 EI 選幾個點
                    get3.append(t[0])
                    get1_3.append(t[1])
                get2.append(get3)
                get1_2.append(get1_3)
            get1.append(get2)
            get1_1.append(get1_2)
        getPoints.append(get1)
        getRatio.append(get1_1)
    return getPoints, getRatio

# 計算 MSE
def result_main2_3(iteration):
    models = [GP.Branching_Kriging, GP.Branching_GP]
    obj_case = [branching_obj_case1_2, branching_obj_bowl2]
    trainDataPaths = ['./pic/meet1015/data/points15.pt', './pic/meet1015/data/points30.pt']

    modelType = {'shareFactors':1, 'branchingFactors':1, 'branchingFactorsLevels':[3], 'nestedFactors':[1]}
    
    getPoints = []
    for m in models:
        get1 = []
        for obj in obj_case:
            get2 = []
            for path in trainDataPaths:
                trainX = torch.load(path)
                trainY = obj(dataTransfer(trainX, modelType))
                get3 = []
                for i in range(iteration):
                    model = m(**modelType)
                    model.get_obj(obj)
                    model.train_datas(trainX=trainX, trainY=trainY)
                    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
                    GP.train(model, optimizer, 200, show=False)

                    ResultM = Result_class()
                    ResultM.real_plot(model, 1, [-6, 6], show=False)
                    ResultM.real_plot(model, 2, [-6, 6], show=False)
                    ResultM.real_plot(model, 3, [-6, 6], show=False)
                    
                    t = ResultM.get_result_MSE(model)
                    get3.append(t)
                get2.append(get3)
            get1.append(get2)
        getPoints.append(get1)
    return getPoints

# 每個 model 對不同 obj function 不同 init points 個數作圖
def result_main3():
    modelDataType = {'shareFactors':1, 'branchingFactors':1, 'branchingFactorsLevels':[3], 'nestedFactors':[1]}
    
    models = [GP.Branching_Kriging, GP.Branching_GP]
    objFuns = [branching_obj_case1_2, branching_obj_bowl2]
    trainDataPaths = ['./pic/meet1015/data/points15.pt', './pic/meet1015/data/points30.pt']

    ResultM = Result_class()
    for i, mod in enumerate(models):
        for j, objFun in enumerate(objFuns):
            for k, path in enumerate(trainDataPaths):
                trainX = torch.load(path)
                trainY = objFun(dataTransfer(trainX, modelDataType))

                model = mod(**modelDataType)
                model.get_obj(objFun)
                model.train_datas(trainX=trainX, trainY=trainY)
                
                optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
                GP.train(model, optimizer, 50, show=False)

                ResultM = Result_class()
                ResultM.real_plot(model, 1, [-6, 6], show=False)
                ResultM.real_plot(model, 2, [-6, 6], show=False)
                ResultM.real_plot(model, 3, [-6, 6], show=False)

                pathM = 'gb' if i == 0  else 'sb' if i == 1 else None
                pathCase = 'case_noCorellation' if j == 0 else 'case_Corellation' if j == 1 else None
                pathP = 'p15' if k == 0 else 'p30' if k == 1 else None

                ZlimPred = [-6, 6] if j == 0 else [0, 3] if j == 1 else None
                cmap = 'Blues' if k == 0 else 'Reds' if k == 1 else None

                for zi in [1, 2, 3]:
                    predPath = './pic/meet1015/{}/{}/{}/pred_z{}.png'.format(pathCase, pathM, pathP, str(zi))
                    EIPath = './pic/meet1015/{}/{}/{}/EI_z{}.png'.format(pathCase, pathM, pathP, str(zi))
                    ContourEI = './pic/meet1015/{}/{}/{}/ContourEI_z{}.png'.format(pathCase, pathM, pathP, str(zi))
                    loadDataPath = './pic/meet1015/{}/{}/{}/Data_z{}.pt'.format(pathCase, pathM, pathP, str(zi))

                    ResultM.get_plot(model, zi, ZlimPred=ZlimPred, 
                                    filesPred=predPath, filesEI=EIPath, filesContourEI=ContourEI, loadData=loadDataPath, cmap=cmap)

# 得到模型對目標函式，尋找np次EI中第幾次找到最小點，以及該點的值與實際最小的ratio
def fun_train(model, objfun, np):
    W = model.every_points()
    WY = objfun(model.data_transfer(W))
    maI = WY.argmin()
    maw = W[maI, :] # 真實最小值的點
    mav = WY.min()  # 真實最小值

    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))
    GP.train(model, optimizer, 200)
    orignPoints, newPoints = GP.train_new_points(model, optimizer, 50, np)
    newPoints = torch.cat(newPoints, dim=0)
    newPointsY = objfun(model.data_transfer(newPoints))  # 預估的值

    ratioBias = 1 - mav
    ratio = (mav + ratioBias) / (newPointsY.min() + ratioBias)
    for n in range(newPoints.shape[0]):
        c1 = newPoints[n, 1].item() == maw[1].item()
        c2 = torch.sqrt(((newPoints[n, [True, False, True]] - maw[[True, False, True]]) ** 2).sum())
        if c1 and c2 <= 0.05:
            return n, ratio
    return 1000, ratio
