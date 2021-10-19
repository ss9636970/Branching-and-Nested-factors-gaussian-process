import Branching_Models_v2 as GP
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import optim
from matplotlib import pyplot as plt

torch.set_printoptions(precision=5)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.ticker import LinearLocator, FormatStrFormatter

import functions as fcs
import fc_data_branching as fdb

if __name__ == '__main__':
    obj_case = fdb.branching_obj_bowl2
    dataType = {'shareFactors':1, 'branchingFactors':1, 'branchingFactorsLevels':[3], 'nestedFactors':[1]}

    # trainX = fdb.initial_datas()
    # trainX = fdb.initial_datas_LHD(20)
    # torch.save(trainX, './pic/meet1015/data/points20.pt')
    trainX = torch.load('./pic/meet1015/data/points15.pt')
    trainY = obj_case(fdb.dataTransfer(trainX, dataType))

    modelType = 'Branching_Kriging'
    modelDataType = {'shareFactors':1, 'branchingFactors':1, 'branchingFactorsLevels':[3], 'nestedFactors':[1]}

    if modelType == 'Branching_Kriging':
        model = GP.Branching_Kriging(**modelDataType)
        
    elif modelType == 'Branching_GP':
        model = GP.Branching_GP(**modelDataType)
        
    model.get_obj(obj_case)
    model.train_datas(trainX=trainX, trainY=trainY)
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

    GP.train(model, optimizer, 100, show=True)
    orignPoints, newPoints = GP.train_new_points(model, optimizer, 50, 20)

    ResultM = fdb.Result_class()
    path = './pic/meet1015/case_Corellation/real/'
    ResultM.real_plot(model, 1, [0, 3], show=True), 
    ResultM.real_plot(model, 2, [0, 3], show=True), 
    ResultM.real_plot(model, 3, [0, 3], show=True)

    path = './pic/meet1015/case_Corellation/'
    p = 15

    ResultM.get_EIPoints_real_plot(1, trainX=trainX, EIpoints=model.trainX[p:, :],
                               file=path+'realEI_p{}_z1.png'.format(str(p)),
                               xtitle='share factor x', ytitle='nested factor v')
    ResultM.get_EIPoints_real_plot(2, trainX=trainX, EIpoints=model.trainX[p:, :],
                                file=path+'realEI_p{}_z2.png'.format(str(p)),
                                xtitle='share factor x', ytitle='nested factor v')
    ResultM.get_EIPoints_real_plot(3, trainX=trainX, EIpoints=model.trainX[p:, :],
                                file=path+'realEI_p{}_z3.png'.format(str(p)),
                                xtitle='share factor x', ytitle='nested factor v')