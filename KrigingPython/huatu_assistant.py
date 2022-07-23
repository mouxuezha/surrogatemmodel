# this is to draw additional figures for surrogare mode .

from copy import deepcopy
from re import A, X
from turtle import color
from cv2 import exp
import numpy as np 
from Surrogate_01de import Surrugate
from huatu import huatu

class huatu_assistant(Surrugate,huatu):
    def __init__(self) -> None:
        huatu.__init__(self,0)
        Surrugate.__init__(self,case='Rotor67',x_dim=18)
        self.y_limit_l = np.array([0.0])
        self.y_limit_u = np.array([1.0])

    def test_fun(self,x):
        a = 1
        x0 = 0
        c = 0
        y = a*(x-x0)**2 + c 
        return y  
    
    def test_fun2(self,x):
        y=x
        return x
    
    def get_data(self,model = 'limit1'):
        N = 500
        if model == 'limit1':
            pingyi = 0 
            L = 2.0  
            obj_func = self.test_fun2
        elif model == 'limit2':
            pingyi = 0 
            L = 2.0  
            obj_func = self.test_fun
        else:
            raise Exception('MXairfoil: invalid model in get_data')

        dx = L / N
        self.X = np.zeros((N,1),dtype=float)
        self.y_original = np.zeros((N,1),dtype=float)
        self.y_limited = np.zeros((N,1),dtype=float)
        for i in range(N):
            if i == 270:
                print('MXairfoil: debug')
            self.X[i,0]= i * dx + pingyi
            # self.y_original[i,0] = self.test_fun(self.X[i,0])
            self.y_original[i,0] = obj_func(self.X[i,0])
            self.y_limited[i] = self.limit_y(self.y_original[i]*1.0)

    def get_tu(self,model='limit1'):
        weizhi = self.location
        self.set_location(weizhi)
        self.get_data(model=model)
        self.x = [] 
        self.y = []
        self.x.append(self.X)
        self.x.append(self.X)
        self.y.append(self.y_original)
        self.y.append(self.y_limited)
        x_zhou = r'$\it{x}$'
        y_zhou = r'$\it{y}$'
        self.huatu2D_mul2(x_zhou,y_zhou,'duibi_limit','Original','Limited',single_ax=True,title_flag=False)
        
        # vertical line
        x_line = np.array([-1,max(self.X)])
        y_line = np.array([self.y_limit_u[0],self.y_limit_u[0]])
        self.ax.plot(x_line, y_line,linestyle='dashdot',linewidth=1,color='k')
        
        if model == 'limit1':
            self.ax.text(0.3, 1.5, r'$\it{y}=\it{x}$', color='k',fontsize=12,fontproperties='Times New Roman')
        elif model == 'limit2':
            self.ax.text(0.3, 2.3, r'$\it{y}=\it{x}^{2}$', color='k',fontsize=12,fontproperties='Times New Roman')
        # 

        self.ax.text(0, 1.1, r'limitation: $\it{y}=1$', color='k',fontsize=10,fontproperties='Times New Roman')
        self.save_all() 

    def jihanshu(self,theta=1,p=2,x=0):
        # theta and p
        x0 = 0 
        y = exp(-theta*abs(x-x0)**p)
        return y 

    def get_data_kriging(self,p=[0.1,1,2],theta=[0.1,1,10],bianliang='p'):
        self.L = 4.0  
        self.N = 500
        self.dx = self.L / self.N
        self.x = [] 
        self.y = []
        if bianliang == 'p':
            for i in range(len(p)):
                p_single = p[i]
                x_sequence,y_sequence = self.get_data_kriging_single(theta[1],p_single)
                self.x.append(x_sequence)
                self.y.append(y_sequence)
        elif bianliang == 'theta':
            for i in range(len(theta)):
                theta_single = theta[i]
                x_sequence,y_sequence = self.get_data_kriging_single(theta_single,p[2])
                self.x.append(x_sequence)
                self.y.append(y_sequence)  
        else:
            raise Exception('MXairfoil: invalid bianliang in get_data_kriging, G ')          

    def get_data_kriging_single(self,theta,p):
        y_sequence = np.zeros((self.N,1),dtype=float)
        x_sequence = np.zeros((self.N,1),dtype=float)
        pianyi = self.L/2.0
        for i in range(self.N):
            if i == 270:
                print('MXairfoil: debug')
            x_sequence[i] = i * self.dx - pianyi
            # self.y_original[i,0] = self.test_fun(self.X[i,0])
            y_sequence[i] = self.jihanshu(theta,p,x_sequence[i])
        return x_sequence,y_sequence

    def get_tu_kriging(self):
        x_zhou = r'$\it{x}$'
        y_zhou = r'$\it{y}$'
        
        self.get_data_kriging(bianliang='p')
        self.huatu2D_mul2(x_zhou,y_zhou,'duibi_p',r'$\it{P}=0.1$',r'$\it{P}=1$',r'$\it{P}=2$',single_ax=True,title_flag=False)
        # horizontal line
        x_line = np.array([0,0])
        y_line = np.array([0,1])
        self.ax.plot(x_line, y_line,linestyle='dashdot',linewidth=1,color='k')
        self.save_all() 

        # then theta
        self.get_data_kriging(bianliang='theta')
        self.huatu2D_mul2(x_zhou,y_zhou,'duibi_theta',r'$\it{\theta}=0.1$',r'$\it{\theta}=1$',r'$\it{\theta}=10$',single_ax=True,title_flag=False)
        # horizontal line
        x_line = np.array([0,0])
        y_line = np.array([0,1])
        self.ax.plot(x_line, y_line,linestyle='dashdot',linewidth=1,color='k')
        self.save_all()         
if __name__ == '__main__':
    shishi = huatu_assistant()
    
    shishi.get_tu(model='limit1')
    shishi.get_tu(model='limit2')
    shishi.get_tu_kriging() 
