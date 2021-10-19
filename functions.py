from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pyDOE as pde
import numpy as np

# 繪製 3d 圖
def plot3d(X, Y, Z, xtitle=None, ytitle=None, ztitle=None, Zlim=None, cmap=cm.coolwarm):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                        linewidth=0, antialiased=False)

    if xtitle:
        ax.set_xlabel(xtitle)

    if ytitle:
        ax.set_ylabel(ytitle)

    if ztitle:
        ax.set_zlabel(ztitle)
    # Customize the z axis.
    if Zlim:
        ax.set_zlim(Zlim[0], Zlim[1])

    # Add a color bar which maps values to colors.
    position = fig.add_axes([0.15, 0.2, 0.05, 0.5])#位置[左,下,右,上]
    fig.colorbar(surf, shrink=0.5, aspect=5, cax=position)
    # fig.colorbar(surf, shrink=0.5, aspect=5, location='left')

    plt.show()
    return fig

# 繪製等高線圖, trainX 為標記初始化資料的點
def plot_contour_line(X, Y, Z, trainX=None, t=None, xtitle=None, ytitle=None):
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, 3, colors='k')
    ax.clabel(CS, fontsize=9, inline=True)
    if trainX is not None:
        tt = trainX.numpy()
        tt = tt[tt[:,1]==t, :]
        ax.scatter(tt[:, 0], tt[:, 2])

    if xtitle:
        ax.set_xlabel(xtitle)

    if ytitle:
        ax.set_ylabel(ytitle)

    plt.show()
    return fig

# 繪製等高線圖, trainX 為標記初始化資料的點, 多標上不同顏色的點
def plot_contour_line2(X, Y, Z, trainX=None, EIpoints=None, t=None, xtitle=None, ytitle=None, minPoint=None):
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, 2, colors='k')
    ax.clabel(CS, fontsize=9, inline=True)
    if trainX is not None:
        tt = trainX.numpy()
        tt = tt[tt[:,1]==t, :]
        ax.scatter(tt[:, 0], tt[:, 2])

    if EIpoints is not None:
        tt = EIpoints.numpy()
        tt = tt[tt[:,1]==t, :]
        ax.scatter(tt[:, 0], tt[:, 2], c='m')

    if minPoint is not None:
        ax.scatter(minPoint[0], minPoint[1], s=100, marker="D", c='#bcbd22', alpha=0.3)

    if xtitle:
        ax.set_xlabel(xtitle)

    if ytitle:
        ax.set_ylabel(ytitle)

    plt.show()
    return fig

# 繪製等高線圖, trainX 為標記初始化資料的點, 多標上不同顏色的點, 每個點標上他是第幾個取的點
def plot_contour_line3(X, Y, Z, trainX=None, EIpoints=None, t=None, xtitle=None, ytitle=None, minPoint=None, pointsNumber=False):
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, 2, colors='k')
    ax.clabel(CS, fontsize=9, inline=True)
    if trainX is not None:
        tt = trainX.numpy()
        tt = tt[tt[:,1]==t, :]
        ax.scatter(tt[:, 0], tt[:, 2])

    if EIpoints is not None:
        tt = EIpoints.numpy()
        tt = tt[tt[:,1]==t, :]
        ax.scatter(tt[:, 0], tt[:, 2], c='m')

    if minPoint is not None:
        ax.scatter(minPoint[0], minPoint[1], s=100, marker="D", c='#bcbd22', alpha=0.3)

    if pointsNumber:
        EN = EIpoints.shape[0]
        EN = np.arange(1, EN+1)
        tt = EIpoints.numpy()
        EN = EN[tt[:,1]==t]
        tt = tt[tt[:,1]==t, :]
        for i in range(EN.shape[0]):
            label = str(EN[i]) + 'th'
            ax.annotate(label, (tt[i, 0], tt[i, 2]), fontsize=13)

    if xtitle:
        ax.set_xlabel(xtitle)

    if ytitle:
        ax.set_ylabel(ytitle)

    plt.show()
    return fig

#用 LHD 生成資料點，只能生數值 factors 資料
def data_LHD(factors, samples):
    return pde.lhs(factors, samples, 'maximin')
