# Branin function from surrogate modelling book.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import time 

def AckleyFunction(x):
    #implement the function itself
    x = np.array(x)
    x = x.reshape(2,)
    x[0] = x[0] * 32.768*2 - 32.768
    x[1] = x[1] * 32.768*2 - 32.768
    a = 20 
    b = 0.2
    c = 2*np.pi 
    d = x.size
    zhi = -a*np.exp(-b*np.sqrt(1/d*(x**2).sum())) - np.exp(1/d*(np.cos(c*x)).sum()) + a + np.exp(1) 
    zhi = zhi - (21.88100063619471)
    zhi = zhi *(-1)
    zhi = zhi / (21.88100063619471)
    return zhi 

def AckleyFunction2(x):
    chicun = x.shape
    try:
        if chicun[1] != 0 :
            #which means array are inputed. 
            zhi = np.zeros([chicun[0],])
            for i in range(chicun[0]):
                # zhi[i] = BraninFunction([x[i][0],x[i][1]])
                zhi[i] = AckleyFunction([x[i][0],x[i][1]])
    except IndexError  :
        # zhi = BraninFunction(x)
        zhi = AckleyFunction(x)
        zhi = np.array(zhi).reshape(1,)
    return zhi

def BraninFunction(x):
    #implement the function itself
    x = np.array(x)
    pi = np.pi
    x = x.reshape(2,)
    x[0] = x[0] * 15 - 5 
    x[1] = x[1] *15
    zhi = (x[1] - 5.1/4/pi**2*x[1]+5/pi*x[0]-6)**2 + 10 *((1-1/8/pi)*np.cos(x[0])+1) + 5*x[0] 
    zhi = zhi - (-16.457520388653005)
    zhi = zhi / (418.5989847628981- (-16.457520388653005))
    return zhi 

def BraninFunction_normal(x):
    x = np.array(x)
    pi = np.pi
    x = x.reshape(2,)
    x[0] = x[0] * 15 - 5 
    x[1] = x[1] *15
    # zhi = 1/51.5*((x[1] - 5.1/4/pi**2*x[0]**2+5/pi*x[0]-6)**2 + (10-10/8/pi)*np.cos(x[0])-44.81) #this this what paper said 
    zhi = ((x[1] - 5.1/4/pi**2*x[0]**2+5/pi*x[0]-6)**2 + (10-10/8/pi)*np.cos(x[0])-44.81)#this is what pykriging used.
    
    
    ymax = 253.31909601160663
    ymin = -54.40622987907503
    y_normal = (zhi-ymin)/(ymax-ymin)
    return y_normal

def BraninFunction2(x):
    chicun = x.shape
    try:
        if chicun[1] != 0 :
            #which means array are inputed. 
            zhi = np.zeros([chicun[0],])
            for i in range(chicun[0]):
                # zhi[i] = BraninFunction([x[i][0],x[i][1]])
                zhi[i] = BraninFunction_normal([x[i][0],x[i][1]])
    except IndexError  :
        # zhi = BraninFunction(x)
        zhi = BraninFunction_normal(x)
        zhi = np.array(zhi).reshape(1,)
    return zhi 

def ToulanFunction2(x):
    chicun = x.shape
    try:
        if chicun[1] != 0 :
            #which means array are inputed. 
            zhi = np.zeros([chicun[0],])
            for i in range(chicun[0]):
                zhi[i] = ToulanFunction([x[i][0],x[i][1]])
    except IndexError  :
        zhi = ToulanFunction(x)
        zhi = np.array(zhi).reshape(1,)
    return zhi 

def ToulanFunction(x):
    y = [0.3,0.7]
    x = np.array(x)
    x = x.reshape(2,)
    y = np.array(y)
    jvli = (x-y)**2
    # zhi = 1-jvli.sum()*2
    jvli1 = ([0,0]-y)**2
    jvli2 = ([0,1]-y)**2
    jvli3 = ([1,0]-y)**2
    jvli4 = ([1,1]-y)**2
    jvli_max = np.max([jvli1.sum(),jvli2.sum(),jvli3.sum(),jvli4.sum()])
    jvli_norm = jvli.sum()/jvli_max
    canshu = 0.15
    zhi = 1 / (jvli_norm + canshu) * canshu
    return zhi  

def shishiSin(x):
    x = np.array(x)
    x = x.reshape(2,)
    pi = np.pi
    # jvli = (x**2).sum() # [0,2]
    # jvli = jvli-1 #[0,1]
    jvli = x[0] #[0,1]
    jvli = jvli*2 -1 #[-1,1]
    zhi = np.sin(pi*jvli)
    return zhi 

def shishi2D(x):
    y = x *2
    return y 

def huatu_1():
    print('MXairfoil: test Branin funtion ')
    x1 = np.arange(0,1.01,0.01)
    x2 = np.arange(0,1.01,0.01)
    X1,X2 = np.meshgrid(x1,x2)
    Y = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        for j in range(X2.shape[0]):
            Y[j][i] = BraninFunction2(np.array([X1[i][i],X2[j][j]]))
    # then plot.
    # Plot the surface.
    norm = cm.colors.Normalize(vmax=Y.max(), vmin=Y.min())
    fig, ax = plt.subplots()
    cset1 = ax.contourf(
    X1, X2, Y, 60,
    norm=norm,alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x_label = np.arange(0,1.1,0.1)
    ax.set_xticks(x_label)
    ax.set_yticks(x_label)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('Branin')
    plt.colorbar(cset1)
    plt.savefig('C:/Users/106/Desktop/Branin.png',dpi=300)
    plt.show()

def huatu_2():
    print('MXairfoil: test ToulanFunction funtion ')
    x1 = np.arange(0,1.01,0.01)
    x2 = np.arange(0,1.01,0.01)
    X1,X2 = np.meshgrid(x1,x2)
    Y = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        for j in range(X2.shape[0]):
            Y[j][i] = ToulanFunction([X1[i][i],X2[j][j]],[0.7,0.3])
    # then plot.
    # Plot the surface.
    norm = cm.colors.Normalize(vmax=Y.max(), vmin=Y.min())
    fig, ax = plt.subplots()
    cset1 = ax.contourf(
    X1, X2, Y, 60,
    norm=norm,alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x_label = np.arange(0,1.1,0.1)
    ax.set_xticks(x_label)
    ax.set_yticks(x_label)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('circle')
    plt.colorbar(cset1)
    plt.savefig('C:/Users/y/Desktop/Toulan.png',dpi=300)
    plt.show()

def huatu_3():
    print('MXairfoil: test shishiSin funtion ')
    x1 = np.arange(0,1.01,0.01)
    x2 = np.arange(0,1.01,0.01)
    X1,X2 = np.meshgrid(x1,x2)
    Y = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        for j in range(X2.shape[0]):
            Y[j][i] = shishiSin([X1[i][i],X2[j][j]])
    # then plot.
    # Plot the surface.
    norm = cm.colors.Normalize(vmax=Y.max(), vmin=Y.min())
    fig, ax = plt.subplots()
    cset1 = ax.contourf(
    X1, X2, Y, 60,
    norm=norm,alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x_label = np.arange(0,1.1,0.1)
    ax.set_xticks(x_label)
    ax.set_yticks(x_label)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('circle')
    plt.colorbar(cset1)
    plt.savefig('C:/Users/y/Desktop/shishiSin.png',dpi=300)
    plt.show()

def huatu2D(function):
    print('MXairfoil: test  funtion ')
    x1 = np.arange(0,1.01,0.01)
    x2 = np.arange(0,1.01,0.01)
    X1,X2 = np.meshgrid(x1,x2)
    Y = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        for j in range(X2.shape[0]):
            Y[j][i] = function(np.array([X1[i][i],X2[j][j]]))
    # then plot.
    # Plot the surface.
    norm = cm.colors.Normalize(vmax=Y.max(), vmin=Y.min())
    fig, ax = plt.subplots()
    cset1 = ax.contourf(
    X1, X2, Y, 60,
    norm=norm,alpha=0.7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x_label = np.arange(0,1.1,0.1)
    ax.set_xticks(x_label)
    ax.set_yticks(x_label)
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title('shishi')
    plt.colorbar(cset1)
    plt.savefig('C:/Users/106/Desktop/huatu2D.png',dpi=300)
    plt.show()

def ceshi2D(realf,predictf):
    time_start = time.time()
    N=100 
    X_rand = np.random.uniform(0,1,(N,2))
    # y_real =np.array([])
    # y_predict =np.array([])
    y_real = np.zeros((N,))
    y_predict = np.zeros((N,))

    

    for i in range(N):
        y_real[i] = realf(X_rand[i,:])
        y_predict[i] = predictf(X_rand[i,:])

    SE = (y_real - y_predict)**2
    MSE = np.mean(SE)
    time_end = time.time()
    time_cost =time_end-time_start
    print('MXairfoil:MSE='+str(MSE)+' time_cost ='+str(time_cost))
    return MSE


if __name__ =='__main__':
    huatu2D(AckleyFunction2)
    print('huatu2D(AckleyFunction2)')