#get a real dailimoxing
from matplotlib.cbook import strip_math
from numpy.lib.function_base import append
import pyKriging  
from pyKriging.krige import kriging  
from pyKriging.samplingplan import samplingplan
import pickle

import numpy as np
from call_components import call_components
import time 
import os

class Surrugate(object):
    
    def __init__(self):
        print('MXairfoil: Sorrogate model initialized')
        self.location = 'C:/Users/106/Desktop/KrigingPython'
        self.script_folder = 'C:/Users/106/Desktop/EnglishMulu/testCDA1'
        self.matlab_location = 'C:/Users/106/Desktop/EnglishMulu/MXairfoilCDA'
        self.real_obs_space_h = np.array([0.4,-0.2,0.8,8])
        self.real_obs_space_l = np.array([-0.2,-0.4,0.2,3])

        self.diaoyong = call_components(self.script_folder,self.matlab_location)

        self.sp = samplingplan(4)
        self.X = [] 

    def __del__(self):
        del self.diaoyong

    def jisuan(self,X):
        X = self.norm_to_real_state(X)

        self.diaoyong.set_value(X[0],'chi_in')

        self.diaoyong.set_value(X[1],'chi_out')

        self.diaoyong.set_value(X[2],'mxthk')

        self.diaoyong.set_value(X[3],'umxthk')

        #start the calculation
        # print('MXairfoil: debugging. Do not actually call CFD components')
        self.diaoyong.call_matlab()
        self.diaoyong.call_IGG()
        self.diaoyong.call_Turbo()
        self.diaoyong.call_CFView()
        omega,rise = self.diaoyong.get_value()

        #for debug
        # omega = X[0]*X[1]
        # rise = X[2]*X[3]

        rizhi = 'MXairfoil: Surrogate model trained for one step.'+'\n state is '+str(X)
        self.diaoyong.jilu(rizhi)
        return omega,rise

    def jisuan_mul(self,diaoyong,X):
        X = self.norm_to_real_state(X)

        diaoyong.set_value(X[0],'chi_in')

        diaoyong.set_value(X[1],'chi_out')

        diaoyong.set_value(X[2],'mxthk')

        diaoyong.set_value(X[3],'umxthk')

        #start the calculation
        # print('MXairfoil: debugging. Do not actually call CFD components')
        diaoyong.call_matlab()
        diaoyong.call_IGG()
        diaoyong.call_Turbo()
        diaoyong.call_CFView()
        omega,rise = diaoyong.get_value()

        #for debug
        # omega = X[0]*X[1]
        # rise = X[2]*X[3]

        rizhi = 'MXairfoil: Surrogate model trained for one step.'+'\n state is '+str(X)
        diaoyong.jilu(rizhi)
        return omega,rise

    def testfun_omega(self,x):
        chicun = x.shape
        try:
            if chicun[1] != 0 :
                #which means array are inputed. 
                zhi = np.zeros([chicun[0],2])
                for i in range(chicun[0]):
                    zhi[i][0],zhi[i][1] = self.jisuan(x[i])
        except IndexError  :
            # zhi = np.zeros([chicun[0],2])
            zhi = np.zeros([1,2])
            zhi[0][0],zhi[0][1] = self.jisuan(x)
            zhi = np.array(zhi).reshape(2,1)
        return zhi[:,0]

    def testfun_rise(self,x):
        chicun = x.shape
        try:
            if chicun[1] != 0 :
                #which means array are inputed. 
                zhi = np.zeros([chicun[0],2])
                for i in range(chicun[0]):
                    zhi[i][0],zhi[i][1] = self.jisuan(x[i])
        except IndexError  :
            # zhi = np.zeros([chicun[0],2])
            zhi = np.zeros([1,2])
            zhi[0][0],zhi[0][1] = self.jisuan(x)
            zhi = np.array(zhi).reshape(2,1)
            return zhi[1,:]
        return zhi[:,1]

    def train_model(self,N):
        self.X = self.sp.optimallhc(N)
        testfun1 = self.testfun_omega
        y1 = testfun1(self.X)
        testfun2 = self.testfun_rise
        y2 = testfun2(self.X)

        self.k1 = kriging(self.X, y1, testfunction=testfun1, name='simple')  
        self.k1.train()

        self.k2 = kriging(self.X, y2, testfunction=testfun2, name='simple')  
        self.k2.train()

        self.y1 = y1 
        self.y2 = y2 
        return y1,y2

    def train_model_mul(self,N,N_thread):
        location_X = self.location+'/X.pkl'
        try:
            self.X = pickle.load(open(location_X,'rb'))
        except:
            self.X = self.sp.optimallhc(N)
            self.X = (self.X - 0.5)*2
            # self.X = self.get_X_toulan(int(np.round(N**0.25)))
            pickle.dump(self.X,open(location_X,'wb'))
        print('MXairfoil: X generated.')
        testfun1 = self.testfun_omega
        testfun2 = self.testfun_rise
        #get the y and save the y
        y = shishi.get_y_mul(self.X,N_thread)
        location_y = self.location+'/y_CFD.pkl'
        pickle.dump(y,open(location_y,'wb'))

        y1 = y[:,0]        
        y2 = y[:,1]
        self.k1 = kriging(self.X, y1, testfunction=testfun1, name='simple')  
        self.k1.train()

        self.k2 = kriging(self.X, y2, testfunction=testfun2, name='simple')  
        self.k2.train()

        self.y1 = y1 
        self.y2 = y2 
        return y1,y2


    def real_to_norm_state(self,state):
        real_state_bili = ( self.real_obs_space_h - self.real_obs_space_l ) /2 
        real_state_c = ( self.real_obs_space_h + self.real_obs_space_l ) /2
        norm_state = (state - real_state_c) / real_state_bili
        return norm_state
    
    def norm_to_real_state(self,state):
        real_state_bili = ( self.real_obs_space_h - self.real_obs_space_l ) /2 
        real_state_c = ( self.real_obs_space_h + self.real_obs_space_l ) /2
        real_state = state*real_state_bili + real_state_c
        return real_state

    def get_y(self,N,X_part):
        print('MXarifoil: this is thread ',N,' ,begin')
        chicun = X_part.shape
        diaoyong = call_components(self.script_folder,self.matlab_location)
        zhi = np.zeros([chicun[0],2])
        for i in range(chicun[0]):
            zhi[i][0],zhi[i][1] = self.jisuan_mul(diaoyong,X_part[i])
        
        #then save y
        wenjianming = self.location + '/'+str(N)+'.pkl'
        pickle.dump(zhi,open(wenjianming,'wb'))
        print('MXarifoil: this is thread ',N,' ,finished','results are in \n' , wenjianming)
        del diaoyong
    
    def get_y_mul(self,X,N):
        import threading 
        threads = [] 
        nloops=range(N)

        # first divide X into parts, and feed into threads.
        chicun = X.shape
        N_part = round(chicun[0] / N)
        for i in range(N-1):
            X_part = X[N_part*i:N_part*(i+1)]
            t = threading.Thread(target=self.get_y,args=(i,X_part))
            threads.append(t)
        i = i +1 
        X_part = X[N_part*i:chicun[0]]
        t = threading.Thread(target=self.get_y,args=(i,X_part))
        threads.append(t)

        #start the threads 
        for i in nloops:
            time.sleep(i*3)#avoid same file name.
            threads[i].start()
        # waiting for the end of all threads
        for i in nloops:
            threads[i].join()

        #then get the results.
        for i in range(N):
            y_part = pickle.load(open(self.location+'/'+str(i)+'.pkl','rb'))
            if i==0 :
                y = y_part
            else:
                y = np.append(y,y_part,axis=0)
        return y

    def save(self):
        location = self.location

        location_k1 = location+'/k1.pkl'
        pickle.dump(self.k1,open(location_k1,'wb'))

        location_k2 = location+'/k2.pkl'
        pickle.dump(self.k2,open(location_k2,'wb'))

        location_X = location+'/X.pkl'
        pickle.dump(self.X,open(location_X,'wb'))

        location_y1 = location+'/y1.pkl'
        pickle.dump(self.y1,open(location_y1,'wb'))

        location_y2 = location+'/y2.pkl'
        pickle.dump(self.y2,open(location_y2,'wb'))

        print('MXairfoil: successfully saved surrogate model')

    def load(self):
        location = self.location

        location_k1 = location+'/k1.pkl'
        self.k1 = pickle.load(open(location_k1,'rb'))

        location_k2 = location+'/k2.pkl'
        self.k2 = pickle.load(open(location_k2,'rb'))

        location_X = location+'/X.pkl'
        self.X = pickle.load(open(location_X,'rb'))

        location_y1 = location+'/y1.pkl'
        self.y1 = pickle.load(open(location_y1,'rb'))

        location_y2 = location+'/y2.pkl'
        self.y2 = pickle.load(open(location_y2,'rb'))

        print('MXairfoil: successfully saved surrogate model')

    def clear(self):
        location = self.location

        location_k1 = location+'/k1.pkl'
        location_k2 = location+'/k2.pkl'
        location_X = location+'/X.pkl'
        location_y1 = location+'/y1.pkl'
        location_y2 = location+'/y2.pkl'

        try:
            os.remove(location_k1)
            os.remove(location_k2)
            # os.remove(location_X)
            os.remove(location_y1)
            os.remove(location_y2)
        except:
            print('MXairfoil: something wrong in clearing surrogate file')

        
        try:
            self.diaoyong.clear_all(self.script_folder,self.matlab_location)
        except:
            print('MXairfoil: something wrong in clearing CFD file')

        print('MXairfoil: finish executing clear process. En Taro XXH')

    def get_X_toulan(self,N):
        #get X in a uniform way, without any lhc or something.
        # X1 = np.zeros((N,4))
        dim = 4 
        X1 = np.zeros(N)
        X2 = np.ones(N)
        for i in range(N):
            X1[i] = i / N
        X = np.zeros((N**dim,dim))
        print('MXairfoil: start generat X')
        for i in range(N**(dim-1)):
            X[(N*(i)):(N*(i+1)),3] = X1
            X[(N*(i)):(N*(i+1)),2] = X1[int(np.floor(i%N))] * X2
            X[(N*(i)):(N*(i+1)),1] = X1[int(np.floor(i/(N))%N)] * X2 
            X[(N*(i)):(N*(i+1)),0] = X1[int(np.floor(i/(N**2))%N)]* X2 


        
        return X
    
    def test_all(self,N):
        #test in sseveral random points, and output the Mean Absolute  Error
        # shishi2  = Surrugate()
        self.load()
        print('MXairfoil: load a Surrogate model, and start to test.')
        
        yy1 = np.zeros([N,2])
        yy2 = np.zeros([N,2])

        location_X_test = self.location+'/X_test.pkl'
        location_yy1 = self.location+'/yy1.pkl'
        location_yy2 = self.location+'/yy2.pkl'
        try:
            X_rand = pickle.load(open(location_X_test,'rb'))
        except:
            X_rand = np.random.uniform(-1,1,(N,4))
            pickle.dump(X_rand,open(location_X_test,'wb'))

        try:
            yy1 = pickle.load(open(location_yy1,'rb'))
            flag_yy1 = 1 
        except:
            flag_yy1 = 0
            yy1 = np.random.uniform(0,1,(N,2))

        try:
            yy2 = pickle.load(open(location_yy2,'rb'))
            flag_yy2 = 1 
        except:
            flag_yy2 = 0
            yy2 = np.random.uniform(0,1,(N,2))


        if flag_yy1 == 0:
            y = shishi.get_y_mul(X_rand,5)
            for i in range(N):
                yy1[i][0]=y[i,0]
                yy2[i][0]=y[i,1]

        
        for i in range(N):
            yy1[i][1] = self.k1.predict(X_rand[i,:])
            yy2[i][1] = self.k2.predict(X_rand[i,:])



        if flag_yy1 == 0:
            pickle.dump(yy1,open(location_yy1,'wb'))
        if flag_yy2 == 0:
            pickle.dump(yy2,open(location_yy2,'wb'))

        #then calculate the MAE
        MAE1 = np.sum(np.abs(yy1[:,0] - yy1[:,1])) / N

        MAE2 = np.sum(np.abs(yy2[:,0] - yy2[:,1])) / N

        print('MXairfoil: test  Surrogate in Mean Absolute  Error','\n MAE1 = ' , MAE1 , '\n MAE2 = ' , MAE2 )

        

        #then,plot 
        #do some visualization.
        import matplotlib.pyplot as plt
        shijian = time.strftime("%Y-%m-%d", time.localtime())
        # plt.xlabel(r'$y_1^real$') 
        # plt.ylabel(r'$y_1^predict$') 
        fig = plt.figure(0)
        plt.xlabel(r'$ \omega_{real}$',fontsize=15) 
        plt.ylabel(r'$\omega_{predict}$',fontsize=15) 
        plt.title('Total pressure loss coefficient')
        marker = '.'
        plt.plot(yy1[:,0],yy1[:,0],marker,color = 'red',label = "marker='{0}'".format(marker)) 
        plt.plot(yy1[:,0],yy1[:,1],marker,color = 'green',label = "marker='{0}'".format(marker))
        plt.legend(('real', 'predict'),loc='upper center', shadow=False) 
        # plt.show()
        wenjianming_omega = self.location + '/omega'+shijian+'.png'
        plt.savefig(wenjianming_omega, dpi=750)
        plt.close(0)

        #then plot the rise
        fig = plt.figure(1)
        plt.xlabel(r'$ Rise_{real}$',fontsize=15) 
        plt.ylabel(r'$ Rise_{predict}$',fontsize=15) 
        plt.title('Static Pressure Rise')
        marker = '.'
        plt.plot(yy2[:,0],yy2[:,0],marker,color = 'red',label = "marker='{0}'".format(marker)) 
        plt.plot(yy2[:,0],yy2[:,1],marker,color = 'green',label = "marker='{0}'".format(marker))
        plt.legend(('real', 'predict'),loc='upper center', shadow=False) 
        # plt.show()
        wenjianming_omega = self.location + '/rise'+shijian+'.png'
        plt.savefig(wenjianming_omega, dpi=750)
        plt.close(1)


if __name__ =='__main__':
    print('MXairfoil: test the Surrogate')
    total_time_start = time.time()
    shishi = Surrugate()
    # shishi.clear()
    flag = 2
    if flag ==0 :
        shishi.load()
        X_rand = np.random.uniform(-1,1,(4,))
        y1_predict2 = shishi.k1.predict(X_rand)
        y1 = shishi.testfun_omega(X_rand)
        
        y2_predict2 = shishi.k2.predict(X_rand)
        y2 = shishi.testfun_rise(X_rand)
        print('MXairfoil: test saving and loading Surrogate','\n y1[0] = ' , y1[0] , '\n its predict = ' , y1_predict2 ,'\n y2[0] = ' , shishi.y2[0] , '\n its predict = ' , y2_predict2)
    elif flag == -1:
        shishi.clear()
    elif flag == 1:
        y1,y2 = shishi.train_model_mul(4**4,10)
        # execute a simple test.
        y1_predict = shishi.k1.predict(shishi.X[0])
        y2_predict = shishi.k2.predict(shishi.X[0])
        print('MXairfoil: test the trained Surrogate','\n y1[0] = ' , y1[0] , '\n its predict = ' , y1_predict ,'\n y2[0] = ' , y2[0] , '\n its predict = ' , y2_predict)

        shishi.save()

        shishi2  = Surrugate()
        shishi2.load()
        y1_predict2 = shishi2.k1.predict(shishi2.X[0])
        y2_predict2 = shishi2.k2.predict(shishi2.X[0])
        print('MXairfoil: test saving and loading Surrogate','\n y1[0] = ' , y1[0] , '\n its predict = ' , y1_predict2 ,'\n y2[0] = ' , y2[0] , '\n its predict = ' , y2_predict2)
    elif flag == 2:
        shishi.test_all(50)

    total_time_end = time.time()
    total_time_cost = total_time_end - total_time_start
    print('MXairfoil: total time cost ='+str(total_time_cost))
    print('MXairfoil: finish a surrogate model related process. En Taro XXH!')



