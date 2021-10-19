import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import random
from scipy.stats import norm

def actFun(x):
    t1 = (x >= 0.) + 0.
    t2 = (-1) * (t1 - 1.)
    output1 = (x + 1) * t1
    output2 = torch.exp(x) * t2
    output = (output1 + output2).clamp(max=30.)
    return output

class Branching_Kriging(nn.Module):
    def __init__(self, shareFactors, branchingFactors, branchingFactorsLevels, nestedFactors):
        super(Branching_Kriging, self).__init__()
        self.shareFactors = shareFactors
        self.branchingFactors = branchingFactors
        self.branchingFactorsLevels = branchingFactorsLevels
        self.nestedFactors = nestedFactors
        
        self.init_parameters()

    def init_parameters(self):
        self.alpha = Parameter(torch.nn.init.normal_(torch.ones(1, self.shareFactors), mean=1.0, std=0.5))
        self.theta = Parameter(torch.nn.init.normal_(torch.ones(1, self.branchingFactors), mean=1.0, std=0.5))

        self.noiseSigma = torch.tensor([1e-6]).view(1, -1)

        nestedFactorsPara = []
        for i in range(self.branchingFactors):
            a, b = self.nestedFactors[i], self.branchingFactorsLevels[i] 
            nestedFactorsPara.append(Parameter(torch.nn.init.normal_(torch.ones(a, b), mean=1.0, std=0.5)))

        self.gamma = nn.ParameterList(nestedFactorsPara)

    def print_parameters(self):
        print('alpha:', self.alpha)
        print('theta:', self.theta)
        print('noiseSigma:', self.noiseSigma)
        print('gamma:', sep=' ')
        for i in self.gamma:
            print(i)

    def get_obj(self, fun):
        self.objF = fun

    def data_transfer(self, W):
        a, b, c = self.shareFactors, self.branchingFactors, sum(self.nestedFactors)
        outputs = [W[:, :a], W[:, a:a + b], W[:, a + b:a + b + c]]
        return outputs

    def train_datas(self, trainX, trainY=None):
        self.trainX = trainX
        if trainY is not None:
            self.trainY = trainY
        else:
            self.trainY = self.objF(self.dataTransfer(trainX))

    def kernel_F(self, W1, W2):
        W1, W2 = self.data_transfer(W1), self.data_transfer(W2)
        X1, X2 = W1[0], W2[0]
        Z1, Z2 = W1[1], W2[1]
        V1, V2 = W1[2], W2[2]

        n1, n2 = X1.shape[0], X2.shape[0]
        
        P1, P2 = X1.repeat(1, n2).view(-1, self.shareFactors), X2.repeat(n1, 1)
        temp = (-1) * (actFun(self.alpha) * (P1 - P2) ** 2).sum(dim=1)
        shareKernel = temp

        P1, P2 = Z1.repeat(1, n2).view(-1, self.branchingFactors), Z2.repeat(n1, 1)
        temp = ((P1 != P2) + 0.) * actFun(self.theta)
        branchingKernel = (-1) * temp.sum(dim=1)

        nestedNum = sum(self.nestedFactors)
        N1, N2 = V1.repeat(1, n2).view(-1, nestedNum), V2.repeat(n1, 1)
        temp = ((N1 - N2) ** 2)
        common = (P1 == P2) + 0.
        levels = (P1 * common).type(torch.long)

        gammaMs = []
        for branchingF_i in range(len(self.gamma)):
            gammaM = self.gamma[branchingF_i]
            gammaM = torch.cat([torch.zeros(gammaM.shape[0], 1), gammaM], dim=1)
            l = levels[:, branchingF_i]
            gammaM = gammaM[:, l].T
            gammaMs.append(gammaM)
        gammaM = actFun(torch.cat(gammaMs, dim=1))
        
        nestedKernel = (-1) * (temp * gammaM).sum(dim=1)
        outputs = torch.exp(shareKernel + branchingKernel + nestedKernel).view(n1, n2)
        return outputs

    def MLE(self):
        n = self.trainX.shape[0]
        Kyy = self.kernel_F(self.trainX, self.trainX) + self.noiseSigma * torch.eye(n)

        KyyInverse = torch.inverse(Kyy)
        
        ones = torch.ones(n, 1)
        mu = (ones.T.mm(KyyInverse).mm(self.trainY)) / (ones.T.mm(KyyInverse).mm(ones))
        sigma = (self.trainY - ones * mu).T.mm(KyyInverse).mm(self.trainY - ones * mu) / n
        Kyydeterminant = torch.det(Kyy)

        if sigma < 0.:
            sigma = torch.exp(sigma)
        else:
            sigma = sigma + 1
        
        if Kyydeterminant < 0.:
            Kyydeterminant = torch.exp(Kyydeterminant)
        else:
            Kyydeterminant = Kyydeterminant + 1

        loss = n * torch.log(sigma+1e-0) + torch.log(Kyydeterminant+1e-0) #+ ((self.trainY - ones * mu).T.mm(KyyInverse).mm(self.trainY - ones * mu)) / sigma
        return loss

    def pred(self, W):
        with torch.no_grad():
            n = self.trainX.shape[0]
            Kyy = self.kernel_F(self.trainX, self.trainX) + self.noiseSigma * torch.eye(n)
            KyyInverse = torch.inverse(Kyy)

            ones = torch.ones(n, 1)
            mu = (ones.T.mm(KyyInverse).mm(self.trainY)) / (ones.T.mm(KyyInverse).mm(ones))

            Kfy = self.kernel_F(W, self.trainX)
            pred = mu + (Kfy.mm(KyyInverse).mm(self.trainY - ones * mu))            
        return pred

    def pred_MSE(self, W):
        with torch.no_grad():
            n = self.trainX.shape[0]
            Kyy = self.kernel_F(self.trainX, self.trainX) + self.noiseSigma * torch.eye(n)
            KyyInverse = torch.inverse(Kyy)

            ones = torch.ones(n, 1)
            mu = (ones.T.mm(KyyInverse).mm(self.trainY)) / (ones.T.mm(KyyInverse).mm(ones))
            sigma = (self.trainY - ones * mu).T.mm(KyyInverse).mm(self.trainY - ones * mu) / n

            Kfy = self.kernel_F(W, self.trainX)

            selfCorrelation = torch.diag(self.kernel_F(W, W)).view(-1, 1)

            term2 = torch.diag(Kfy.mm(KyyInverse).mm(Kfy.T)).view(-1, 1)
            term3 = ((1 - ones.T.mm(KyyInverse).mm(Kfy.T)) / (ones.T.mm(KyyInverse).mm(ones))).T
            predMSE = sigma * (selfCorrelation - term2 + term3)
            p = (predMSE > 0) + 0.
            predMSE = predMSE * p
        return predMSE

    def EI(self, W):
        with torch.no_grad():
            predMSE = self.pred_MSE(W)
            predMSE = torch.sqrt(predMSE)
            yMin = torch.min(self.trainY)
            yPred = self.pred(W)

            isNotZero = (predMSE > 0.00001) + 0.

            predMSE = predMSE + (-1) * (isNotZero - 1)

            c = yMin - yPred
            value = c / predMSE
            nPdf = norm.pdf(value, loc=0, scale=1)
            nPdf = torch.tensor(nPdf, dtype=torch.float)
            nCdf = norm.cdf(value, loc=0, scale=1)
            nCdf = torch.tensor(nCdf, dtype=torch.float)

            outputs = (c * nCdf + predMSE * nPdf) * isNotZero
        return outputs

    # 生成所有可能的點，準備丟到 EI 去選點
    def every_points(self):      # 以 0.05 為區間
        shareFactors, branchingFactors,  = self.shareFactors, self.branchingFactors
        branchingFactorsLevels, nestedFactors = self.branchingFactorsLevels, sum(self.nestedFactors)
        snFactors = shareFactors + nestedFactors
        grids = [np.arange(0, 1.05, 0.05)] * snFactors
        grids = np.meshgrid(*grids)
        grids = list(map(torch.tensor, grids))
        for i, j in enumerate(grids):
            grids[i] = torch.unsqueeze(j, dim=0)
        grids = torch.cat(grids, dim=0).view(snFactors, -1).T

        grids2 = []
        for i in branchingFactorsLevels:
            grids2.append(np.arange(1, i+1, 1))

        grids2 = np.meshgrid(*grids2)
        grids2 = list(map(torch.tensor, grids2))

        for i, j in enumerate(grids2):
            grids2[i] = torch.unsqueeze(j, dim=0)
        grids2 = torch.cat(grids2, dim=0).view(branchingFactors, -1).T
        
        n1 = grids.shape[0]
        n2 = grids2.shape[0]
        grids = grids.repeat(n2, 1)
        grids2 = grids2.repeat(1, n1).view(-1, branchingFactors) 
        
        p = snFactors + branchingFactors
        index = list(range(0, shareFactors)) + list(range(snFactors, p)) + list(range(shareFactors, snFactors))
        grids = torch.cat([grids, grids2], dim=1)[:, index]
        return grids.type(torch.float)

# 使用EI選點並加入 model 資料當中
    def EI_get_points(self):
        W = self.every_points()
        EICurve = self.EI(W)
        EIIndex = EICurve.argmax().item()
        maxPoint = W[[EIIndex], :]
        y = self.objF(self.data_transfer(maxPoint))
        self.trainX = torch.cat([self.trainX, maxPoint], dim=0)
        self.trainY = torch.cat([self.trainY, y], dim=0)
        return maxPoint

# 學長的主要 model
class Branching_GP(Branching_Kriging):
    def __init__(self, shareFactors, branchingFactors, branchingFactorsLevels, nestedFactors):
        super(Branching_GP, self).__init__(shareFactors, branchingFactors, branchingFactorsLevels, nestedFactors)

    def init_parameters(self):
        self.alpha = Parameter(torch.nn.init.normal_(torch.ones(1, self.shareFactors), mean=1.0, std=0.5))
        self.noiseSigma = torch.tensor([1e-6]).view(1, -1)
        
        nestedFactorsPara = []
        for i in range(self.branchingFactors):
            a, b = self.nestedFactors[i], self.branchingFactorsLevels[i] 
            nestedFactorsPara.append(Parameter(torch.nn.init.normal_(torch.ones(a, b), mean=1.0, std=0.5)))
        self.gamma = nn.ParameterList(nestedFactorsPara)

    def print_parameters(self):
        print('alpha:', self.alpha)
        print('noiseSigma:', self.noiseSigma)
        print('gamma:', sep=' ')
        for i in self.gamma:
            print(i)

    # def adjust_parameters(self):
    #     with torch.no_grad():
    #         self.alpha.clamp_(min=0.0)
    #         self.noiseSigma.clamp_(min=1e-6)
            
    #         for i, m in enumerate(self.gamma):
    #             m.clamp_(min=0.0)

    def kernel_F(self, W1, W2):
        W1, W2 = self.data_transfer(W1), self.data_transfer(W2)
        X1, X2 = W1[0], W2[0]
        Z1, Z2 = W1[1], W2[1]
        V1, V2 = W1[2], W2[2]

        n1, n2 = X1.shape[0], X2.shape[0]
        
        P1, P2 = X1.repeat(1, n2).view(-1, self.shareFactors), X2.repeat(n1, 1)
        temp = (-1) * (actFun(self.alpha) * (P1 - P2) ** 2).sum(dim=1)
        shareKernel = temp

        P1, P2 = Z1.repeat(1, n2).view(-1, self.branchingFactors), Z2.repeat(n1, 1)

        nestedNum = sum(self.nestedFactors)
        N1, N2 = V1.repeat(1, n2).view(-1, nestedNum), V2.repeat(n1, 1)
        temp = ((N1 - N2) ** 2)
        common = (P1 == P2) + 0.
        commonAll = (common.sum(dim=1) == self.branchingFactors) + 0.
        levels = (P1 * common).type(torch.long)

        gammaMs = []
        for branchingF_i in range(len(self.gamma)):
            gammaM = self.gamma[branchingF_i]
            gammaM = torch.cat([torch.zeros(gammaM.shape[0], 1), gammaM], dim=1)
            l = levels[:, branchingF_i]
            gammaM = gammaM[:, l].T
            gammaMs.append(gammaM)
        gammaM = actFun(torch.cat(gammaMs, dim=1))
        
        nestedKernel = (-1) * (temp * gammaM).sum(dim=1)
        outputs = (torch.exp(shareKernel + nestedKernel) * commonAll).view(n1, n2)
        return outputs

def train(model, optimizer, iterations, show=False):
    for i in range(iterations):
        optimizer.zero_grad()
        loss = model.MLE()
        # if torch.isnan(loss).item():
        #     break
        
        if show:
            print('{} step -- Loss: {}'.format(str(i), str(loss.item())))
        
        loss.backward()
        optimizer.step()
        # model.adjust_parameters()

def train_new_points(model, optimizer, iterations, pointsN):
    orignPoints = model.trainX
    newPoints = []
    for i in range(pointsN):
        newPoints.append(model.EI_get_points())
        train(model, optimizer, iterations)
    return orignPoints, newPoints
