#get a real dailimoxing
from gc import collect
from pyexpat import model
from re import X
import pyKriging  
from pyKriging.krige import kriging  
from pyKriging.samplingplan import samplingplan
import pickle

import numpy as np
from call_components import call_components
from transfer import transfer
import time 
import time_out 
import os
import shutil

import matplotlib.pyplot as plt
from matplotlib import cm

from parameters import parameters
from multiprocessing import Process
import threading 

import sys


class Surrugate(object):
    
    def __init__(self,**kargs):
        print('MXairfoil: Sorrogate model initialized')
        # self.diaoyong = call_components(self.script_folder,self.matlab_location) # no need for real calculating here, disable the jisuan.
        if 'case' in kargs:
            if kargs['case'] == 'NACA65':
                self.set_locations_NACA65()
                self.N_points = 0 
            elif kargs['case'] == 'CDA1':
                self.set_locations_CDA1()
                self.N_points = 0 
            elif kargs['case'] == 'Rotor67':
                self.set_locations_Rotor67()
            elif kargs['case'] == 'Rotor37':
                self.set_locations_Rotor37()
            else:
                raise Exception('MXairfoil: invalid case for surrogate_01de, G!')
            self.case = kargs['case']
        else:
            raise Exception('MXairfoil: no case setted in surrogate_01de,G!')

        if 'x_dim' in kargs:
            self.x_dim = kargs['x_dim']
        else:
            self.x_dim = 4 

        self.sp = samplingplan(self.x_dim)
        self.X = [] 
        self.N_correct = 0 
        
        self.N_test = 0 
        self.MAE = np.zeros(3)
        self.model = 'train'

        self.N_fail =0 

    def set_locations_CDA1(self):        
        if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
            #which means in my diannao
            self.location = 'C:/Users/y/Desktop/KrigingPython'
            self.script_folder = 'C:/Users/y/Desktop/temp/testCDA1'
            self.matlab_location = 'C:/Users/y/Desktop/temp/MXairfoilCDA'
        elif os.environ['COMPUTERNAME'] == 'DESKTOP-132CR84' :
            # which means in new working zhan. D:\XXHdatas\EnglishMulu
            # raise Exception('MXairfoil: no location setted, G')
            self.location = 'D:/XXHcode/KrigingPython'
            self.script_folder = 'D:/XXHdatas/EnglishMulu/testCDA1'
            self.matlab_location = 'D:/XXHdatas/EnglishMulu/MXairfoilCDA'            
        else:
            # which means in 106 server   
            self.location = 'C:/Users/106/Desktop/KrigingPython'
            self.script_folder = 'C:/Users/106/Desktop/temp/testCDA1'
            self.matlab_location = 'C:/Users/106/Desktop/temp/MXairfoilCDA'
        
        # self.real_obs_space_h = np.array([0.4,-0.2,0.8,8])
        # self.real_obs_space_l = np.array([-0.2,-0.4,0.2,3])
        # self.real_obs_space_h = np.array([0.4,-0.1,0.8,8])
        # self.real_obs_space_l = np.array([0.3,-0.5,0.2,3])
        self.real_obs_space_h = np.array([0.35,-0.22,0.55,8])
        self.real_obs_space_l = np.array([0.25,-0.38,0.35,5])

    def set_locations_NACA65(self):
        if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
            #which means in my diannao
            self.location = 'C:/Users/y/Desktop/KrigingPython'
            self.script_folder = 'C:/Users/y/Desktop/temp/testNACA65'
            self.matlab_location = 'C:/Users/y/Desktop/temp/MXairfoilNACA65'
        elif os.environ['COMPUTERNAME'] == 'DESKTOP-132CR84' :
            # which means in new working zhan. D:\XXHdatas\EnglishMulu
            # raise Exception('MXairfoil: no location setted, G')
            self.location = 'D:/XXHcode/KrigingPython'
            self.script_folder = 'D:/XXHdatas/EnglishMulu/testNACA65'
            self.matlab_location = 'D:/XXHdatas/EnglishMulu/MXairfoilNACA65'   
        else:
            # which means in 106 server   
            self.location = 'C:/Users/106/Desktop/KrigingPython'
            self.script_folder = 'C:/Users/106/Desktop/EnglishMulu/testNACA65'
            self.matlab_location = 'C:/Users/106/Desktop/EnglishMulu/MXairfoilNACA65'
        self.real_obs_space_h = np.array([0.49,-0.22,0.055,0.45])
        self.real_obs_space_l = np.array([0.37,-0.34,0.045,0.35])

    def __del__(self):
        try:
            del self.diaoyong
        except:
            print('MXairfoil: no diaoyong object here, exit directly')

    def jisuan(self,X):
        X = self.norm_to_real_state(X)
        if 0 :
            omega = X[0]
            rise = X[1]
            print('MXairfoil: attention, jisuan is debugging!')
            return omega,rise

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

    # @time_out.time_out(200, time_out.callback_func)
    def jisuan_mul(self,diaoyong,X):
        X_debug = X
        X = self.norm_to_real_state(X)
        if 0 :
            omega = X[0]
            rise = X[1]
            print('MXairfoil: attention, jisuan_mul is debugging!')
            return omega,rise

        diaoyong.set_value(X[0],'chi_in')

        diaoyong.set_value(X[1],'chi_out')

        diaoyong.set_value(X[2],'mxthk')

        diaoyong.set_value(X[3],'umxthk')

        #start the calculation
        # print('MXairfoil: debugging. Do not actually call CFD components')
        cishu = 1
        omega = np.zeros(cishu)
        rise = np.zeros(cishu)
        turn = np.zeros(cishu)
        diaoyong.call_matlab()
        diaoyong.call_IGG()
        for i in range(cishu):
            diaoyong.call_Turbo()
            diaoyong.call_CFView()
            # omega[i],rise[i],turn[i] = diaoyong.get_value_new()
            try:
                if diaoyong.done ==1:
                    # which means something rong
                    return 0,0,0
                # omega[i],rise[i] = diaoyong.get_value()
                omega[i],rise[i],turn[i] = diaoyong.get_value_new()
            except:
                os.system('pause')
                return 0,0,0
        #for debug
        # omega = X[0]*X[1]
        # rise = X[2]*X[3]
        # rizhi = 'MXairfoil: struggle to find out bug..'+'\n state is '+str(X)+'\n '+diaoyong.result_folder + '\n' + str(omega)+'   ' + str(rise)  + '\n Its X is  ' +str(X_debug)
        # diaoyong.jilu(rizhi)
        self.jindu.calculate_1ci()
        self.jindu.check_progress()
        return omega.mean(),rise.mean(),turn.mean()

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

    def testfun_turn(self,x):
        chicun = x.shape
        try:
            if chicun[1] != 0 :
                #which means array are inputed. 
                zhi = np.zeros([chicun[0],3])
                for i in range(chicun[0]):
                    zhi[0][0],zhi[0][1],zhi[0][2] = self.jisuan(x[i])
        except IndexError  :
            # zhi = np.zeros([chicun[0],2])
            zhi = np.zeros([1,3])
            zhi[0][0],zhi[0][1],zhi[0][2] = self.jisuan(x)
            zhi = np.array(zhi).reshape(3,1)
            return zhi[2,:]
        return zhi[:,2]

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
        self.N_points = N
        self.jindu = record_progress(N_points = self.N_points)
        location_X = self.location+'/X.pkl'
        try:
            self.X = pickle.load(open(location_X,'rb'))
        except:
            self.X = self.sp.optimallhc(N)
            pickle.dump(self.X,open(location_X,'wb'))
        print('MXairfoil: X generated.')
        testfun1 = self.testfun_omega
        testfun2 = self.testfun_rise
        testfun3 = self.testfun_turn

        #get the y and save the y
        location_y = self.location+'/y_CFD.pkl'
        try:
            y = pickle.load(open(location_y,'rb'))
        except:
            y = self.get_y_mul(self.X,N_thread)
            pickle.dump(y,open(location_y,'wb'))

        self.y1 = y[:,0]   
        self.y2 = y[:,1]
        self.y3 = y[:,2]

        flag = 1
        while flag >0 :
            flag = self.check_existing_data()

        y1 = self.y1    
        y2 = self.y2 
        y3 = self.y3

        #define and train the kriging model.
        self.k1 = kriging(self.X, y1, testfunction=testfun1, name='simple')  
        self.k1.train()

        self.k2 = kriging(self.X, y2, testfunction=testfun2, name='simple')  
        self.k2.train()

        self.k3 = kriging(self.X, y3, testfunction=testfun3, name='simple')  
        self.k3.train()

        return y1,y2,y3

    def real_to_norm_state(self,state):
        # this is change into [0,1]
        real_state_bili = ( self.real_obs_space_h - self.real_obs_space_l )
        norm_state = (state - self.real_obs_space_l) / real_state_bili
        return norm_state
    
    def norm_to_real_state(self,state):
        # this is change from [0,1] to real 
        real_state_bili = ( self.real_obs_space_h - self.real_obs_space_l )
        real_state = state*real_state_bili + self.real_obs_space_l
        return real_state

    def get_y(self,N,X_part):
        chicun = X_part.shape
        if chicun[0]==0:
            #which means X is empty here.
            print('MXairfoil: empty X are fed into some threads, warning')
            return 

        zhi = np.zeros([chicun[0],3])

        # this is for calculate 
        diaoyong = call_components(self.script_folder,self.matlab_location,case=self.case) 
        # rizhi = '\n\n\nMXairfoil:this is thread '+str(N)+' ,begin, \n X_part is' + str(X_part) +'\n result folder:'+ diaoyong.result_folder + '\n matlab location:' + diaoyong.matlab_location
        # diaoyong.jilu(rizhi)  

        for i in range(chicun[0]):
            zhi[i][0],zhi[i][1],zhi[i][2] = self.jisuan_mul(diaoyong,X_part[i])
            # try:
            #     zhi[i][0],zhi[i][1],zhi[i][2] = self.jisuan_mul(diaoyong,X_part[i])
            # except:
            #     rizhi = '\n\n\nMXairfoil:this is thread '+str(N)+'\n result folder:'+ diaoyong.result_folder + '\n matlab location:' + diaoyong.matlab_location + '\n***********************\nRunning time out\n***********************\n'
            #     diaoyong.jilu(rizhi)  
            #     zhi[i][0] = 0
            #     zhi[i][1] = 0
            #     zhi[i][2] = 0
            #     os.system('pause')

        del diaoyong

        # # this is for debug;
        # print('MXairfoil: attention! get_y is running in debug model')
        # for i in range(chicun[0]):
        #     zhi[i][0] = X_part[i][0] 
        #     zhi[i][1] = X_part[i][1] 
        
        #then save y
        wenjianming = self.location + '/'+str(N)+'.pkl'
        pickle.dump(zhi,open(wenjianming,'wb'))
        try:
            print('MXarifoil: this is thread ',N,' ,finished','results are in \n' , wenjianming,'X_part is ',X_part,'\n\n\n')
            self.jindu.check_progress()
        except OSError:
            print('MXarifoil: this is thread '+str(N)+', find an OSError here')
        # time.sleep(0.1)
        # self.clear_process()       
    
    def get_y_mul(self,X,N):
        # import threading 
        threads = [] 
        nloops=range(N)

        # first divide X into parts, and feed into threads.
        chicun = X.shape
        N_part = round(chicun[0] / N)

        for i in range(N-1):
            X_part = X[N_part*i:N_part*(i+1)]
            t = threading.Thread(target=self.get_y,args=(i,X_part))
            threads.append(t)
        #     print(i)
        # print(i)
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
            # y_part = pickle.load(open(self.location+'/'+str(i)+'.pkl','rb'))
            try:
                y_part = pickle.load(open(self.location+'/'+str(i)+'.pkl','rb'))
                if i==0 :
                    y = y_part
                else:
                    y = np.append(y,y_part,axis=0)    
            except:
                print('MXairfoil: cannnot get the pkl of thread '+str(i)+',  warning')
                y = np.array([])
        return y

    def get_y_mul2(self,X,N):
        # it is said that multiprocessing is more powerful.
        processes = [] 
        nloops=range(N)

        # first divide X into parts, and feed into threads.
        chicun = X.shape
        N_part = round(chicun[0] / N)

        for i in range(N-1):
            X_part = X[N_part*i:N_part*(i+1)]
            t = Process(target=self.get_y,args=(i,X_part))
            processes.append(t)
        #     print(i)
        # print(i)
        i = i +1 
        X_part = X[N_part*i:chicun[0]]
        t = Process(target=self.get_y,args=(i,X_part))
        processes.append(t)

        #start the threads 
        for i in nloops:
            time.sleep(i*3) # avoid same file name.
            processes[i].start()
        # waiting for the end of all threads
        for i in nloops:
            processes[i].join()

        #then get the results.
        for i in range(N):
            # y_part = pickle.load(open(self.location+'/'+str(i)+'.pkl','rb'))
            try:
                y_part = pickle.load(open(self.location+'/'+str(i)+'.pkl','rb'))
                if i==0 :
                    y = y_part
                else:
                    y = np.append(y,y_part,axis=0)    
            except:
                print('MXairfoil: cannnot get the pkl of thread '+str(i)+',  warning')
                y = np.array([])
        return y

    def save(self):
        location = self.location
        if self.case == 'Rotor67':
            dim = self.N_model
            for i in range(dim):
                location_ki = location + '/k'+str(i+1)+'.pkl'
                pickle.dump(self.kx_list[i],open(location_ki,'wb'))
                location_yi = location+'/y'+str(i+1)+'.pkl'
                pickle.dump(self.yx_list[i],open(location_yi,'wb'))
        else:
            location_k1 = location+'/k1.pkl'
            pickle.dump(self.k1,open(location_k1,'wb'))

            location_k2 = location+'/k2.pkl'
            pickle.dump(self.k2,open(location_k2,'wb'))

            location_k3 = location+'/k3.pkl'
            pickle.dump(self.k3,open(location_k3,'wb'))

            location_X = location+'/X.pkl'
            pickle.dump(self.X,open(location_X,'wb'))

            location_y1 = location+'/y1.pkl'
            pickle.dump(self.y1,open(location_y1,'wb'))

            location_y2 = location+'/y2.pkl'
            pickle.dump(self.y2,open(location_y2,'wb'))

            location_y3 = location+'/y3.pkl'
            pickle.dump(self.y3,open(location_y3,'wb'))

        print('MXairfoil: successfully saved surrogate model')

    def load(self):
        location = self.location    
        if self.case == 'Rotor67':
            dim = self.N_model
            self.kx_list=[]
            self.yx_list=[]
            for i in range(dim):
                location_ki = location + '/k'+str(i+1)+'.pkl'
                ki = pickle.load(open(location_ki,'rb'))
                self.kx_list.append(ki)
                location_yi = location+'/y'+str(i+1)+'.pkl'
                yi = pickle.load(open(location_yi,'rb'))
                self.yx_list.append(yi)
        else:

            location_k1 = location+'/k1.pkl'
            self.k1 = pickle.load(open(location_k1,'rb'))

            location_k2 = location+'/k2.pkl'
            self.k2 = pickle.load(open(location_k2,'rb'))

            location_k3 = location+'/k3.pkl'
            self.k3 = pickle.load(open(location_k3,'rb'))

            location_X = location+'/X.pkl'
            self.X = pickle.load(open(location_X,'rb'))

            location_y1 = location+'/y1.pkl'
            self.y1 = pickle.load(open(location_y1,'rb'))

            location_y2 = location+'/y2.pkl'
            self.y2 = pickle.load(open(location_y2,'rb'))

            location_y3 = location+'/y3.pkl'
            self.y3 = pickle.load(open(location_y3,'rb'))

        print('MXairfoil: successfully loaded surrogate model')

    def clear(self):
        location = self.location
        if self.case == 'Rotor67':
            dim = self.N_model
            for i in range(dim):
                location_ki = location + '/k'+str(i+1)+'.pkl'
                location_yi = location+'/y'+str(i+1)+'.pkl'
                location_yyi = location+'/yy'+str(i+1)+'.pkl'
                try:
                    os.remove(location_ki)
                    os.remove(location_yi) 
                    os.remove(location_yyi) 
                except:
                    print('MXairfoil: something wrong in clearing surrogate file')
            location_ygezhong = []  
            location_ygezhong.append(location+'/y_CFD.pkl')
            location_ygezhong.append(location+'/y_list.pkl')
            location_ygezhong.append(location+'/y_rep.pkl')
            location_ygezhong.append(location+'/y_test_CFD.pkl')
            location_ygezhong.append(location+'/y_test_list.pkl')

            for i in range(100):
                location_threadi = location + '/'+str(i)+'.pkl'
                try:
                    os.remove(location_threadi)
                except:
                    print('MXairfoil: something wrong when clearing surrogate file')
        else:
            location_k1 = location+'/k1.pkl'
            location_k2 = location+'/k2.pkl'
            location_k3 = location+'/k3.pkl'
            location_X = location+'/X.pkl'
            location_y1 = location+'/y1.pkl'
            location_y2 = location+'/y2.pkl'
            location_y3 = location+'/y3.pkl'
            location_yCFD = location+'/y_CFD.pkl'


            try:
                os.remove(location_k1)
                os.remove(location_k2)
                os.remove(location_k3)
                os.remove(location_X)
                os.remove(location_y1)
                os.remove(location_y2)
                os.remove(location_y3)
                os.remove(location_yCFD)
            except:
                print('MXairfoil: something wrong in clearing surrogate file')

        
        try:
            self.diaoyong.clear_all(self.script_folder,self.matlab_location)
        except:
            print('MXairfoil: something wrong in clearing CFD file')

        location_Ctemp = 'C:/tmp'
        try:
            shutil.rmtree(location_Ctemp)
            print('MXairfoil: clear C temp file for numeca successfully')
        except:
            print('MXairfoil: clear C temp file for numeca fail, in surrogate_01de')
        
        try:
            self.diaoyong.del_gai()
        except:
            print('MXairfoil: something wrong in clearing CFD file')

        print('MXairfoil: finish executing clear process. En Taro XXH')

    def test_all(self,N,N_thread=72,**kargs):
        #test in sseveral random points, and output the Mean Absolute  Error
        # shishi2  = Surrugate()
        
        self.N_test = N 
        self.N_points= N 
        if 'self_model' in kargs:
            self.model = kargs['self_model']
            if self.model =='demo180':
                self.set_demo180()
        else:
            self.model = 'test'
        if 'load_model' in kargs:
            # this is to decide which kriging model are used.
            load_model = kargs['load_model'] # 'exist' for using existing kriging models.
        else:
            load_model = 'import' # 'import' for load models from outside.
        
        if 'X_model' in kargs:
            # this is to decide which X are used.
            X_model = kargs['X_model'] # 'X_train' and 'X_rand', and 'X_merge'
        else:
            X_model = 'X_rand'

        if load_model == 'import':
            self.load()
            print('MXairfoil: load a Surrogate model, and start to test.')
        elif load_model == 'exist':
            print('MXairfoil: existing model using.')
        else:
            raise Exception('MXairfoil: invalid load_model in test_all')

        self.jindu = record_progress(N_points = self.N_test*self.N_steps_per_point)


        
        if self.case == 'Rotor67':
            location_yyi_list = [] 
            for i in range(self.N_model):
                location_yyi = self.location+'/yy'+str(i+1)+'.pkl'
                location_yyi_list.append(location_yyi)
        else:
            location_yy1 = self.location+'/yy1.pkl'
            location_yy2 = self.location+'/yy2.pkl'
            location_yy3 = self.location + '/yy3.pkl'
            location_yyi_list = [location_yy1,location_yy2,location_yy3]

        if X_model == 'X_rand':
            # load X_rand
            location_X_test = self.location+'/X_test.pkl'
            location_y = self.location+'/y_test_CFD.pkl'
        elif X_model == 'X_train':
            location_X_test = self.location+'/X.pkl'
            location_y = self.location+'/y_CFD.pkl'
        elif X_model == 'X_merge':
            location_X_test = [self.location+'/X_test.pkl' , self.location+'/X.pkl']
            location_y = [self.location+'/y_test_CFD.pkl' , self.location+'/y_CFD.pkl']

        else:
            raise Exception('MXairfoil: invalid X_model')


        try:
            X_rand = self.load_mul(location_X_test)
            N = len(X_rand)
            self.N_test = N 
            self.N_points= N             
        except:
            if self.case=='Rotor67':
                X_rand = np.random.uniform(0,1,(N,self.x_dim))
                # X_rand = self.sp.optimallhc(N)
                
            else:
                X_rand = np.random.uniform(0,1,(N,4))           
            self.save_mul(X_rand,location_X_test)

        if self.case == 'Rotor67':
            yyx_list = [] 
            for i in range(self.N_model):
                try:
                    # yyi = pickle.load(open(location_yy1,'rb'))
                    yyi = self.load_mul(location_yyi_list[i])
                    flag_yyi = 1 
                except:
                    flag_yyi = 0
                    yyi = np.random.uniform(0,1,(N,2))
                yyx_list.append(yyi)

            if flag_yyi == 0:
                #get the y and save the y
                try:
                    # y = pickle.load(open(location_y,'rb'))
                    y = self.load_mul(location_y)
                except:
                    # recalculate CFD only when need.
                    y = self.get_y_3D_mul2(X_rand,N_thread,model=self.model)
                    # pickle.dump(y,open(location_y,'wb'))
                    self.save_mul(y,location_y)
                    
                for i in range(N):
                    for j in range(self.N_model):
                        yyx_list[j][i][0] = y[i,j]

            for i in range(N):
                for j in range(self.N_model):
                    yyx_list[j][i][1] = self.kx_list[j].predict(X_rand[i,:])

            if flag_yyi == 0:
                for i in range(self.N_model):
                    # pickle.dump(yyx_list[i],open(location_yyi_list[i],'wb'))
                    self.save_mul(yyx_list[i],location_yyi_list[i])

            self.MAE = [] 
            self.Rfang = []
            for i in range(self.N_model):
                chazhii = np.abs(yyx_list[i][:,0] - yyx_list[i][:,1]) 
                MAEi = np.sum(chazhii) / N
                Rfangi = self.calculate_R2(yyx_list[i][:,0],yyx_list[i][:,1]) # yy1[:,0] is y_real
                print('MXairfoil: test  Surrogate, index = '+str(i)+'\n MAE = ' +str( MAEi) + ' , Rfangi = '+ str(Rfangi))
                self.MAE.append(MAEi)
                self.Rfang.append(Rfangi)
            from huatu import huatu
            tu = huatu(0)
            tu.set_location(self.location)
            tu.plot_surrogate_3D(yyx_list)

        else:
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

            try:
                yy3 = pickle.load(open(location_yy3,'rb'))
                flag_yy3 = 1 
            except:
                flag_yy3 = 0
                yy3 = np.random.uniform(0,1,(N,3))

            try:
                yy4 = pickle.load(open(location_yy4,'rb'))
                flag_yy4 = 1 
            except:
                flag_yy4 = 0
                yy4 = np.random.uniform(0,1,(N,3))




            if flag_yy1 == 0:
                #get the y and save the y
                location_y = self.location+'/y_test_CFD.pkl'
                try:
                    y = pickle.load(open(location_y,'rb'))
                except:
                    # recalculate CFD only when need.
                    if self.case=='Rotor67':
                        y = self.get_y_3D_mul2(X_rand,self.N_thread,model ='test')
                    else:
                        y = self.get_y_mul(X_rand,5)
                    pickle.dump(y,open(location_y,'wb'))
                    
                for i in range(N):
                    yy1[i][0]=y[i,0]
                    yy2[i][0]=y[i,1]
                    yy3[i][0]=y[i,2]
                    if self.case=='Rotor67':
                        yy4[i][0]=y[i,3]

            # check the data, even it is raw generated 
            # y1_corrected, y2_corrected,y3_coreected = self.check_existing_data(y1 = yy1[:,0],y2=yy2[:,0],y3=yy3[:,0],X=X_rand)
            # # since python transfer variables(unfundamental type) by referece rather than by value, these two lines are unnecessary.
            # yy1[:,0]=y1_corrected
            # yy2[:,0]=y2_corrected

            
            for i in range(N):
                #always re-calculate predict
                yy1[i][1] = self.k1.predict(X_rand[i,:])
                yy2[i][1] = self.k2.predict(X_rand[i,:])
                yy3[i][1] = self.k3.predict(X_rand[i,:])
                if self.case=='Rotor67':
                    yy4[i][1] = self.k4.predict(X_rand[i,:])


            if flag_yy1 == 0:
                pickle.dump(yy1,open(location_yy1,'wb'))
            if flag_yy2 == 0:
                pickle.dump(yy2,open(location_yy2,'wb'))
            if flag_yy3 == 0:
                pickle.dump(yy3,open(location_yy3,'wb'))
            if self.case=='Rotor67':
                if flag_yy4 == 0:
                    pickle.dump(yy4,open(location_yy4,'wb'))            

            #then calculate the MAE
            chazhi1 = np.abs(yy1[:,0] - yy1[:,1]) 
            chazhi1_index = np.argwhere(chazhi1>(chazhi1.max()-0.00001))

            chazhi2 = np.abs(yy2[:,0] - yy2[:,1]) 
            chazhi2_index = np.argwhere(chazhi2>(chazhi2.max()-0.00001))

            chazhi3 = np.abs(yy3[:,0] - yy3[:,1]) 
            chazhi3_index = np.argwhere(chazhi3>(chazhi3.max()-0.00001))

            MAE1 = np.sum(chazhi1) / N

            MAE2 = np.sum(chazhi2) / N

            MAE3 = np.sum(chazhi3) / N

            print('MXairfoil: test  Surrogate in Mean Absolute  Error','\n MAE1 = ' , MAE1 , '\n MAE2 = ' , MAE2,'\n MAE3 = ' , MAE3 )
            self.MAE[0]=MAE1
            self.MAE[1]=MAE2
            self.MAE[2]=MAE3


            # 2021-8-31 21:12:59 add something to calculate R^2
            Rfang1 = self.calculate_R2(yy1[:,0],yy1[:,1]) # yy1[:,0] is y_real
            Rfang2 = self.calculate_R2(yy2[:,0],yy2[:,1])
            Rfang3 = self.calculate_R2(yy3[:,0],yy3[:,1])
            print('MXairfoil: test  Surrogate in R^2','\n Rfang1 = ' , Rfang1 , '\n Rfang2 = ' , Rfang2,'\n Rfang3 = ' , Rfang3 )        


            if self.case=='Rotor67':
                from huatu import huatu
                y_list = [yy1,yy2,yy3,yy4]
                tu = huatu(0)
                tu.set_location(self.location)
                tu.plot_surrogate_3D(y_list)
            else: 
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

                #then plot the turn
                fig = plt.figure(2)
                plt.xlabel(r'$ Turn_{real}$',fontsize=15) 
                plt.ylabel(r'$ Turn_{predict}$',fontsize=15) 
                plt.title('Flow Angle Turn')
                marker = '.'
                plt.plot(yy3[:,0],yy3[:,0],marker,color = 'red',label = "marker='{0}'".format(marker)) 
                plt.plot(yy3[:,0],yy3[:,1],marker,color = 'green',label = "marker='{0}'".format(marker))
                plt.legend(('real', 'predict'),loc='upper center', shadow=False) 
                # plt.show()
                wenjianming_omega = self.location + '/turn'+shijian+'.png'
                plt.savefig(wenjianming_omega, dpi=750)
                plt.close(2)
        self.auto_record()

    def load_mul(self,location_list):
        # load many values and merge them.
        if type(location_list) == list :
            N = len(location_list)
            value = pickle.load(open(location_list[0],'rb'))
            for i in range(N-1):
                value_i = pickle.load(open(location_list[i+1],'rb'))
                value = np.append(value,value_i,axis=0)
        elif type(location_list) == str:
            value = pickle.load(open(location_list,'rb'))
        return value

    def save_mul(self,value,location_list):
        # this is to match up with self.load_mul()
        if type(location_list) == list :
            # 
            print('MXairfoil: more than one location, value will not be saved')
        elif type(location_list) == str:
            zhi = pickle.dump(value,open(location_list,'wb'))
        return zhi        

    def visual_2D(self,x_name,y_name):
        shijian = time.strftime("%Y-%m-%d", time.localtime())
        print('MXairfoil: test visual_2D for surrogate model\n'+shijian)
        diaoyong  = call_components(self.script_folder,self.matlab_location,case=self.case) 
        X=np.zeros((parameters.get_number(),))
        for canshu in parameters:
            value_location = diaoyong.matlab_location+'/input/'+self.case+'/'+canshu.name+'.txt'
            X[canshu.value] = diaoyong.get_value2(value_location)
        X = self.real_to_norm_state(X)
        self.test_point(X)

        x1 = np.arange(0,1.01,0.01)
        x2 = np.arange(0,1.01,0.01)
        X1,X2 = np.meshgrid(x1,x2)
        Y1 = np.zeros(X1.shape)
        Y2 = np.zeros(X2.shape)
        Y3 = np.zeros(X2.shape)
        
        # get some data for trainning ANN .
        for i in range(X1.shape[1]):
            for j in range(X2.shape[0]):
                X[parameters[x_name].value] = X1[i][i]
                X[parameters[y_name].value] = X2[j][j]
                Y1[j][i] = self.k1.predict(X) # ji not ij, for i corresponds to shape[1]
                Y2[j][i] = self.k2.predict(X)
                Y3[j][i] = self.k3.predict(X)
                # Y1[i][j],Y2[i][j] = self.jisuan(X)
        
        # save the data for further using:
        wenjianing_X1 = self.location + '/visual2DX1.pkl'
        wenjianing_X2 = self.location + '/visual2DX2.pkl'
        wenjianing_Y1 = self.location + '/visual2DY1.pkl'
        wenjianing_Y2 = self.location + '/visual2DY2.pkl'
        wenjianing_Y3 = self.location + '/visual2DY3.pkl'

        pickle.dump(X1,open(wenjianing_X1,'wb'))
        pickle.dump(X2,open(wenjianing_X2,'wb'))
        pickle.dump(Y1,open(wenjianing_Y1,'wb'))
        pickle.dump(Y2,open(wenjianing_Y2,'wb'))
        pickle.dump(Y3,open(wenjianing_Y3,'wb'))

        x_name = parameters[x_name]
        y_name = parameters[y_name]
        #then plot, omega 
        norm = cm.colors.Normalize(vmax=Y1.max(), vmin=Y1.min())
        fig, ax = plt.subplots()
        cset1 = ax.contourf(
        X1, X2, Y1, 60,
        norm=norm,alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        x_label = np.arange(0,1.1,0.1)
        ax.set_xticks(x_label)
        ax.set_yticks(x_label)
        ax.set_xlabel(parameters.get_equation(x_name))
        ax.set_ylabel(parameters.get_equation(y_name))
        biaoti_omega = r'$\omega$'
        ax.set_title(biaoti_omega)
        plt.colorbar(cset1)
        plt.savefig(self.location+'/visual2Domega'+shijian+'.png',dpi=300)
        # plt.show()

        # then plot rise. copying code is very stupid, so do I . 
        norm = cm.colors.Normalize(vmax=Y2.max(), vmin=Y2.min())
        fig, ax = plt.subplots()
        cset1 = ax.contourf(
        X1, X2, Y2, 60,
        norm=norm,alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        x_label = np.arange(0,1.1,0.1)
        ax.set_xticks(x_label)
        ax.set_yticks(x_label)
        ax.set_xlabel(parameters.get_equation(x_name))
        ax.set_ylabel(parameters.get_equation(y_name))
        # ax.set_xlabel(r'$\chi_{in}$')
        # ax.set_ylabel(r'$\chi_{out}$')
        biaoti_rise = r'$Rise$'
        ax.set_title(biaoti_rise)
        plt.colorbar(cset1)
        plt.savefig(self.location+'/visual2Drise'+shijian+'.png',dpi=300)
        # plt.show()

        # then plot turn. copying code is very stupid, so do I . 
        norm = cm.colors.Normalize(vmax=Y3.max(), vmin=Y3.min())
        fig, ax = plt.subplots()
        cset1 = ax.contourf(
        X1, X2, Y3, 60,
        norm=norm,alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        x_label = np.arange(0,1.1,0.1)
        ax.set_xticks(x_label)
        ax.set_yticks(x_label)
        ax.set_xlabel(parameters.get_equation(x_name))
        ax.set_ylabel(parameters.get_equation(y_name))
        biaoti_turn = r'$Turn$'
        ax.set_title(biaoti_turn)
        plt.colorbar(cset1)
        plt.savefig(self.location+'/visual2Dturn'+shijian+'.png',dpi=300)
        # plt.show()

    def clear_process(self):
        mingling1 = 'taskkill /F /IM rm.exe'
        print(mingling1)
        mingling2 = 'taskkill /F /IM cmd.exe'
        print(mingling2)
        try:
            os.system(mingling1)
            os.system(mingling2)
        except:
            print('MXairfoil: nothing to clear.')

    def load_compare(self):
        # load tow groups of data to compare. trying to get to know the different.
        locationB = 'C:/Users/106/Desktop/KrigingPython/backup/backup_compare' # datas assumed to be right
        locationA = 'C:/Users/106/Desktop/KrigingPython/backup'

        self.location = locationA 
        self.load()
        X_A = self.X
        y1_A = self.y1
        y2_A = self.y2
        k1_A = self.k1
        # fig1_A = self.k1.plot()
        # plt.show()
        # self.k1.saveFigure(locationA+'compare.png')


        # y_compare = self.get_y_mul(X_A[0:20],5) 
        # y_compare2 = shishi.get_y_mul(X_A[0:20],5) # exactly the same.

        self.location = locationB 
        self.load()
        X_B = self.X
        y1_B = self.y1
        y2_B = self.y2
        k1_b = self.k1

        MAE_X = np.sum(np.abs(X_A - X_B))
        MAE_y1 = np.sum(np.abs(y1_A - y1_B))
        MAE_y2 = np.sum(np.abs(y2_A - y2_B))
        # MAE_y_compare = np.sum(np.abs(y_compare[:,0] - y1_B))
        print('MXairfoil: compareing two datasets.')

    def xianzhi_cheat(self,y):
        # limiter to get better ssurrogate model. 2 sigma.
        mean_y = np.mean(y)
        sigma_y = np.std(y)
        low_y = mean_y-3*sigma_y
        high_y = mean_y + 3*sigma_y

        high_gg_index = np.argwhere(y>high_y)
        low_gg_index = np.argwhere(y<low_y)
        
        if len(high_gg_index)>0 :
            y[high_gg_index.astype(int)] = high_y
        if len(low_gg_index)>0:
            y[low_gg_index.astype(int)] = low_y

        self.N_correct = self.N_correct + len(high_gg_index) + len(low_gg_index)

        return y 

    def check_existing_data(self, **kargs):
        # this is for 2d case.
        if len(kargs)>0:
            model = 1
            y1 = kargs['y1']
            y2 = kargs['y2']
            y3 = kargs['y3']
            # y3 = kargs['y3']
            X = kargs['X']
        else:
            model = 0 
            y1 = self.y1
            y2 = self.y2
            y3 = self.y3
            X = self.X
        # this is to check the data before trainning. to avoid wasting time.
        y1_gg_index = np.argwhere(y1<0.008) # this is (n,1)
        y1_gg_index = np.append(y1_gg_index,np.argwhere(y1>0.2),axis=0)

        y2_gg_index = np.argwhere(y2<1)
        y2_gg_index = np.append(y2_gg_index,np.argwhere(y2>1.1),axis=0)

        y3_gg_index = np.argwhere(self.y3<3)
        y3_gg_index = np.append(y3_gg_index,np.argwhere(self.y3>50),axis=0)
        # y3_gg_index = np.array([]).reshape(0,1)



        panju = len(y1_gg_index) + len(y2_gg_index) +len(y3_gg_index)

        gg_index = np.append(y1_gg_index,np.append(y2_gg_index,y3_gg_index,axis=0),axis=0)
        gg_index = np.unique(gg_index)
        gg_index = gg_index.reshape(len(gg_index),)
        X_gg = X[gg_index.astype(int)]

        diaoyong  = call_components(self.script_folder,self.matlab_location,case=self.case) 
        if panju>0:
            rizhi = 'MXairfoil: there must be something wrong in prepared data'+'\n y1_gg_index = '+str(y1_gg_index)+'\n y2_gg_index = '+str(y2_gg_index)+'\n y3_gg_index = '+str(y3_gg_index)+'\nproblemable X is '+ str(X_gg)
            diaoyong.jilu(rizhi)

        # correcte the wrong point in single thread
        
        for i in range(len(X_gg)):
            y1_correct, y2_correct,y3_correct = self.jisuan_mul(diaoyong,X_gg[i])
            y1[gg_index[i].astype(int)] = y1_correct
            y2[gg_index[i].astype(int)] = y2_correct
            y3[gg_index[i].astype(int)] = y3_correct
            if model == 0:
                self.y1[gg_index[i].astype(int)] = y1_correct
                self.y2[gg_index[i].astype(int)] = y2_correct
                self.y3[gg_index[i].astype(int)] = y3_correct
            self.N_correct = self.N_correct + 1

        if len(X_gg) ==0:
            print('MXairfoil: y corrected done')
        else:
            print('MXairfoil: y corrected, for '+str(gg_index))

        # it is not enough, using something zuobi to deal outlier
        y1_cheat = self.xianzhi_cheat(y1)
        y2_cheat = self.xianzhi_cheat(y2)
        y3_cheat = self.xianzhi_cheat(y3)

        if model == 0 :
            self.y1 = y1_cheat
            self.y2 = y2_cheat
            self.y3 = y3_cheat
            return len(X_gg)
        elif model == 1 :
            return y1_cheat, y2_cheat, y3_cheat

        # return len(X_gg)

    def visual_3D_debug(self):
        # this is to visualization then trainning dataset, to know where the question is.
        changdu = len(self.X)
        x = self.X[:,0]
        y = self.X[:,1]
        z = self.y2
        ax = plt.axes(projection='3d')
        ax.plot3D(x,y,z,linestyle ='None',marker = '.',color='k')
        plt.savefig(self.location+'/visual3Ddebug.png',dpi=300)
        plt.show()

    def auto_record(self):
        # auto record the configuration in a single txt.
        print('MXairfoil: Auto record the configuration and result in txt')
        shijian = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
        wenjianming = self.location + '\configuration.txt'
        wenjian = open(wenjianming,'w')
        wenjian.write('MXairfoil:configuration recorded. En Taro XXH!\n')
        wenjian.write('\nN_points='+str(self.N_points))
        wenjian.write('\nN_test='+str(self.N_test))
        wenjian.write('\nMAE='+str(self.MAE))
        wenjian.write('\nR^2='+str(self.Rfang))
        wenjian.write('\nN_correct='+str(self.N_correct))
        try:
            wenjian.write('\npmin='+str(self.pmin))
            wenjian.write('\nthetamax'+str(self.thetamax))
        except:
            pass
        try:
            wenjian.write('\nxishu_limit'+str(self.xishu_limit))
        except:
            pass

        # wenjian.write('\nreal_obs_space_h='+str(self.real_obs_space_h))
        # wenjian.write('\nreal_obs_space_l='+str(self.real_obs_space_l))
        wenjian.write('\nfinished at:'+shijian)
        wenjian.close()

    def test_point(self,state):
        # state_normal is [0,1] here
        
        if  state[3]>3:
            # which means state is real state 
            state_surrogate = self.real_to_norm_state(state)
        else:
            # which means state is surrogate state.
            state_surrogate = state
        omega_predict = shishi.k1.predict(state_surrogate)
        rise_predict = shishi.k2.predict(state_surrogate)
        turn_predict = shishi.k3.predict(state_surrogate)
        print('MXairfoil: input state ='+str(state)+'\nsurrogate state = ' +str(state_surrogate)+'\n predicted omega: '+ str(omega_predict)+'\n predicted omega: '+ str(rise_predict)+ '\n predicted turn: '+ str(turn_predict)) 

    def test_repeatability(self):
        # this is to test the repeatability of CFD parts.
        location_X = self.location+'/X.pkl'
        self.X = pickle.load(open(location_X,'rb'))
        print('MXairfoil: X loaded.')
        x_test = self.X[38]
        # x_test = np.array([0.99,0.01,0.99,0.01])
        self.N_points  = 24 
        N_thread = 8
        self.jindu = record_progress(N_points = self.N_points)
        location_X = self.location+'/X_rep.pkl'
        x_all = np.array([x_test])
        try:
            self.X = pickle.load(open(location_X,'rb'))
        except:
            for i in range(self.N_points-1):
                x_all = np.append(x_all,x_test.reshape(1,4),axis=0)
            self.X = x_all
            pickle.dump(self.X,open(location_X,'wb'))
        print('MXairfoil: X generated for testing repeatability.')

        #get the y and save the y
        location_y = self.location+'/y_rep.pkl'
        try:
            y = pickle.load(open(location_y,'rb'))
        except:
            y = self.get_y_mul(self.X,N_thread)
            pickle.dump(y,open(location_y,'wb'))

        self.y1 = y[:,0]   
        self.y2 = y[:,1]
        self.y3 = y[:,2] 

        for i in range(3):
            norml_err = np.sqrt(np.var(y[:,i])) /np.mean(y[:,i])
            print('MXairfoil: this is dimension '+str(i) + ', the variance is ' + str(norml_err))  

    def test_data(self):
        # this is to check the calculated CFD data, to get outlier point.
        location_X = self.location+'/X.pkl'
        self.X = pickle.load(open(location_X,'rb'))
        print('MXairfoil: X loaded.')

        #get the y and save the y
        location_y = self.location+'/y_CFD.pkl'
        y = pickle.load(open(location_y,'rb'))

        self.y1 = y[:,0]   
        self.y2 = y[:,1]
        self.y3 = y[:,2]

        for i in range(3):
            y_i = y[:,i]
            index_max = np.argwhere(y_i==np.max(y_i))
            index_min = np.argwhere(y_i==np.min(y_i))
            x_index_max = self.X[index_max]
            x_index_min = self.X[index_min] 
            y_mean = np.mean(y_i)
            print('MXairfoil: finished one dimension.')

    def calculate_R2(self,y_real , y_predict):
        #2021-8-31 21:14:55 this is to calculate R2, since MAE is not so good.
        fenzi =np.sum((y_predict - y_real )**2)
        pingjun = np.mean(y_real)
        fenmu = np.sum((y_real - pingjun)**2) 
        Rfang = 1 - fenzi / fenmu
        return Rfang

    # 3D=============================================
    def set_locations_Rotor67(self):
        if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
            #which means in my diannao
            self.location = 'C:/Users/y/Desktop/KrigingPython'
            self.script_folder = 'C:/Users/y/Desktop/EnglishMulu/testRotor67'
            self.matlab_location = 'C:/Users/y/Desktop/EnglishMulu/MXairfoilRotor67'
            self.N_thread = 4
        elif os.environ['COMPUTERNAME'] == 'DESKTOP-132CR84' :
            # which means in new working zhan. D:\XXHdatas\EnglishMulu
            # raise Exception('MXairfoil: no location setted, G')
            self.location = 'D:/XXHcode/KrigingPython'
            self.script_folder = 'D:/XXHdatas/EnglishMulu/testRotor67'
            self.matlab_location = 'D:/XXHdatas/EnglishMulu/MXairfoilRotor67' 
            self.N_thread = 72  
        else:
            # which means in 106 server   
            self.location = 'C:/Users/106/Desktop/KrigingPython'
            self.script_folder = 'C:/Users/106/Desktop/EnglishMulu/testRotor67'
            self.matlab_location = 'C:/Users/106/Desktop/EnglishMulu/MXairfoilRotor67'
        self.N_steps_per_point = 40 
        self.zhuansu = -16173
        Pout_range = 30000 
        # self.Pout_min = 128591.6 - Pout_range/2
        self.Pout_d = 700

        self.Pout_min = 128591.6 - Pout_range/3*2 + 500
        
        self.get_Pout_distribution(Pout_range)
        # self.N_thread = 72
        self.Pout = 128591.6 
        self.massflow_deviation_threshold=0.5
        self.massflow_0 = 34.6
        self.massflow_threshold = 20
        self.efficiency_threshold=0.87
        self.N_model = 7
        real_obs_space_h = np.array([0.07,0.14,0.21,0.03,0.06,0.09,0.48,0.15,0.16,0.15,-0.6,-0.02,-0.018,-0.12,0.26,0.7,1,1.15])
        real_obs_space_l = np.array([-0.07,-0.14,-0.21,-0.03,-0.06,-0.09,0.41,0.1,-0.04,-0.05,-0.67,-0.08,-0.08,-0.18,0.18,0.6,0.92,1.05])
        self.bianhuan = transfer(tishi = 0 , dim = 18,real_obs_space_h = real_obs_space_h,real_obs_space_l=real_obs_space_l)
        self.state_original_real = np.array( [0,0,0,0,0,0,0.447895,0.122988,0.064253,0.050306, -0.639794,-0.052001,-0.050454,-0.148836,0.223533,0.656313,0.965142,1.098645]) 
        # y_geometry_single = [massflow_rate_lower,massflow_rate_upper,efficiency_panju,pi_panju,massflow_rate_w,eta_w,pi_w] 
        self.y_limit_l = np.array([0.85,0.93,0.87,1.6,0.875,0.87,1.6])
        self.y_limit_u = np.array([0.99,1.035,0.89,1.68,1.025,0.9,1.67])
        self.xishu_limit = 3
        self.N_limited = 0 

    def set_locations_Rotor37(self):
        if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
            #which means in my diannao
            self.location = 'C:/Users/y/Desktop/KrigingPython'
            self.script_folder = 'C:/Users/y/Desktop/EnglishMulu/testRotor37'
            self.matlab_location = 'C:/Users/y/Desktop/EnglishMulu/MXairfoilRotor37'
        elif os.environ['COMPUTERNAME'] == 'DESKTOP-132CR84' :
            # which means in new working zhan. D:\XXHdatas\EnglishMulu
            # raise Exception('MXairfoil: no location setted, G')
            self.location = 'D:/XXHcode/KrigingPython'
            self.script_folder = 'D:/XXHdatas/EnglishMulu/testRotor37'
            self.matlab_location = 'D:/XXHdatas/EnglishMulu/MXairfoilRotor37'   
        else:
            # which means in 106 server   
            self.location = 'C:/Users/106/Desktop/KrigingPython'
            self.script_folder = 'C:/Users/106/Desktop/EnglishMulu/testRotor37'
            self.matlab_location = 'C:/Users/106/Desktop/EnglishMulu/MXairfoilRotor37'
        self.zhuansu = -17188
        self.Pout_min = 128500  #11800 for left part of points.
        self.Pout_d = 20
        self.N_points = 70
        self.N_thread = 4
        self.Pout = 120000 
        self.massflow_deviation_threshold=0.5
        self.massflow_0 = 20.83
        self.efficiency_threshold=0.87
        self.N_model = 7

    def operating_3D(self):
        # this is to calculate caractristics for one geometry.
        # constant outlet pressure and changable zhuansu.
        self.N_points = 20
        N_thread = 3  
        zhuansu_min = -16173 + 2000
        zhuansu_d = 200

        self.jindu = record_progress(N_points = self.N_points)

        zhuansu = np.zeros([self.N_points,1]) 
        for i in range(self.N_points):
            zhuansu[i][0] = zhuansu_min - zhuansu_d * i 
        jieguo = self.get_y_3D_mul(zhuansu,N_thread)
        print('MXairfoil: finish generating characteristics. En Taro XXH')

    def operating_3D2(self):
        # this is to calculate caractristics for one geometry.
        # constant zhuansu and changable outlet Static_Pressure.
        zhuansu = self.zhuansu
        Pout_min = self.Pout_min
        Pout_d = self.Pout_d

        self.jindu = record_progress(N_points = self.N_points)

        X = np.zeros([self.N_points,2]) 
        for i in range(self.N_points):
            X[i][0] = zhuansu
            X[i][1] = Pout_min + Pout_d * i 

        jieguo = self.get_y_3D_mul(X,self.N_thread)
        print('MXairfoil: finish generating characteristics. En Taro XXH')

    def get_result_3D(self,N,X_part):
        # this is for self.operating3D, and something like this.
        chicun = X_part.shape
        
        if chicun[0]==0:
            #which means X is empty here.
            raise Exception('MXairfoil: empty X are fed into some threads, G!', N)
        elif chicun[1] == self.x_dim:
            print('MXairfoil: get_result_3D for different geometries. so-called trainning')
        elif chicun[1] == 2:
            # operating for one single geometry. multiprocess
            print('MXairfoil: operating for one single geometry. multiprocess')
        else:
            raise Exception('MXairfoil: wrong dimension when get_result_3D, G!')

        zhi = np.zeros([chicun[0],4])

        # this is for calculate 
        diaoyong = call_components(self.script_folder,self.matlab_location,case=self.case) 
        diaoyong.massflow_deviation_threshold = self.massflow_deviation_threshold
        diaoyong.massflow_0 = self.massflow_0
        diaoyong.efficiency_threshold=self.efficiency_threshold
        for i in range(chicun[0]):
            zhi[i][0],zhi[i][1],zhi[i][2] = self.jisuan_3D_mul(diaoyong,X_part[i]) 
        del diaoyong

        #then save y
        wenjianming = self.location + '/'+str(N)+'.pkl'
        pickle.dump(zhi,open(wenjianming,'wb'))
        try:
            print('MXarifoil: this is thread ',N,' ,finished','results are in \n' , wenjianming,'X_part is ',X_part,'\n\n\n')
            self.jindu.check_progress()
        except OSError:
            print('MXarifoil: this is thread '+str(N)+', find an OSError here')
        # time.sleep(0.1)
        # self.clear_process()    

    def get_result_3D_list(self,N,X_part):
        # this is for trainning surrogate model
        chicun = len(X_part)
        if chicun == 0 :
            print('MXairfoil: no X_list for this thread, '+str(N)+', return.')
            return 
        
        zhi = np.zeros([chicun,4])

        # this is for calculate 
        if (self.model == 'rescue'): # or (self.model == 're_calculate'):
            diaoyong = call_components(self.script_folder,self.matlab_location,case=self.case,index=self.N_thread+3)
            diaoyong.clear_all(self.script_folder,self.matlab_location)

        diaoyong = call_components(self.script_folder,self.matlab_location,case=self.case,index=N+1) 
        diaoyong.massflow_deviation_threshold = self.massflow_deviation_threshold
        diaoyong.massflow_0 = self.massflow_0
        diaoyong.efficiency_threshold=self.efficiency_threshold
        for i in range(chicun):
            zhi[i][0],zhi[i][1],zhi[i][2] = self.jisuan_3D_mul2(diaoyong,X_part[i]) 
            # time.sleep(200)
            # print('MXairfoil: debuging, here is thread'+str(N))
        del diaoyong

        #then save y
        wenjianming = self.location + '/'+str(N)+'.pkl'
        pickle.dump(zhi,open(wenjianming,'wb'))
        try:
            print('MXarifoil: this is thread ',N,' ,finished','results are in \n' , wenjianming,'X_part length: ',chicun,'\n\n\n')
            self.jindu.check_progress()
        except OSError:
            print('MXarifoil: this is thread '+str(N)+', find an OSError here')
        # time.sleep(0.1)
        # self.clear_process()    

    def jisuan_3D_mul(self,diaoyong,X):
        # this is for operatinfg_3D and something like this.
        diaoyong.set_zhuansu(X[0])
        
        if len(X)>1:
            diaoyong.set_Pout(X[1])

        diaoyong.call_Turbo() 

        massFlow,piStar,eta_i = diaoyong.get_result_3D(target = self.script_folder)

        self.jindu.calculate_1ci()
        self.jindu.check_progress()

        return massFlow,piStar,eta_i

    def jisuan_3D_mul2(self,diaoyong,X,**kargs):
        # input: X as a list. x[0]: index of geometry point, x[1]: input for geometry generation,input for call_components. x[2]: CFD setting.
        # output: performance of a 3D rotor.
        
        # # debug 
        # if X[0][0] == 232: #67 232
        #     print('MXairfoil: skimming data, now number ' + str(X[0][0]) + ' geometry points.')
        #     pass # this is for debug

        # 0, check
        if self.model == 're_calculate':
            # do not check duplicate 
            pass 
        else:
            flag,location = self.jindu.check_3D_result(index_geometry = X[0][0], index_CFD = X[0][1],model=self.model)
            if flag == True:
                # which means this has been calculated
                massFlow,piStar,eta_i = diaoyong.get_result_3D(resultLocation=location)
                self.jindu.calculate_1ci()
                return massFlow,piStar,eta_i

        index_CFD_previous = X[0][1]-1
        if index_CFD_previous > 10 :
            # first several points are low-risk.
            flag,location_previous = self.jindu.check_3D_result(index_geometry = X[0][0], index_CFD = X[0][1],model=self.model)
            if flag == True :
                # previous point has been calculated.
                # then read and check.
                massFlow,piStar,eta_i = diaoyong.get_result_3D(resultLocation=location_previous)
                if (massFlow <self.massflow_threshold) and (massFlow>0.1):
                    # which means previous point has been G, and not [0,nan,nan]
                    # then just copy previous to now.
                    self.jindu.save_3D_result(location_previous,index_geometry = X[0][0], index_CFD = X[0][1],model=self.model)
                    self.jindu.calculate_1ci()
                    return massFlow,piStar,eta_i
                    pass
                
        
        # 1,build the geometry.
        X_real = self.bianhuan.surrogate_to_real(X[1])
        
        diaoyong.set_value_3D(X_real)

        diaoyong.reset_result_3D() # clean and do not collect

        diaoyong.call_matlab()

        diaoyong.call_AutoGrid5()

        # 2, set CFD
        zhuansu = X[2][0]
        yali = X[2][1]
        diaoyong.set_zhuansu(zhuansu)
        diaoyong.set_Pout(yali)

        # 3,run 
        
        try:
            diaoyong.call_Turbo() 
        except:
            print('MXairfoil: one point has GGed, return zero')
            self.jindu.GG_1ci(X)
            return 0,0,0
        # time.sleep(3)
        massFlow,piStar,eta_i = diaoyong.get_result_3D()
        piStar = float(piStar)
        eta_i = float(eta_i)

        # 4,record jindu.
        self.jindu.calculate_1ci()
        self.jindu.check_progress()
        self.jindu.save_3D_result(diaoyong.script_folder,index_geometry = X[0][0], index_CFD = X[0][1],model=self.model)

        return massFlow,piStar,eta_i

    def get_y_3D_mul(self,X,N):
        # this is for calculate curves of one geometry.
        threads = [] 
        nloops=range(N)

        # first divide X into parts, and feed into threads.
        chicun = X.shape
        N_part = round(chicun[0] / N)

        for i in range(N-1):
            X_part = X[N_part*i:N_part*(i+1)]
            t = threading.Thread(target=self.get_result_3D,args=(i,X_part))
            threads.append(t)
        # print(i)
        # print(i)
        try:
            i = i + 1
        except:
            i=0 
        X_part = X[N_part*i:chicun[0]]
        t = threading.Thread(target=self.get_result_3D,args=(i,X_part))
        threads.append(t)

        #start the threads 
        for i in nloops:
            time.sleep(i*3) #avoid same file name.
            threads[i].start()
        # waiting for the end of all threads
        for i in nloops:
            threads[i].join()

        #then get the results.
        for i in range(N):
            # y_part = pickle.load(open(self.location+'/'+str(i)+'.pkl','rb'))
            try:
                y_part = pickle.load(open(self.location+'/'+str(i)+'.pkl','rb'))
                if i==0 :
                    y = y_part
                else:
                    y = np.append(y,y_part,axis=0)    
            except:
                print('MXairfoil: cannnot get the pkl of thread '+str(i)+',  warning')
                y = np.array([])
        return y

    def get_y_3D_mul2(self,X,N,**kargs):
        # this is for trainning surrogate model.
        
        threads = [] 
        
        if 'model' in kargs:
            # train, test,rescue,test_rep,testfun.
            model = kargs['model']
        else:
            model = self.model

        X_list = self.get_X_list(X)
        if self.case == 'Rotor67':
            X_list = self.jindu.check_3D_X_list(X_list,model=model)
            wanchengdu = self.jindu.check_process_CFD()
            if wanchengdu > 0.99999:
                # finished. then pretend nothing happend
                X_list = self.get_X_list(X)
                if (model == 'train')or((model == 'test')): 
                    N = 1 # get y serially.
        
        nloops=range(N)

        if model == 'train':
            location_X_list = self.location+'/X_list.pkl'
            pickle.dump(X_list,open(location_X_list,'wb'))
        elif model == 'rescue':
            location_X_list = self.jindu.location_X_failed
            pickle.dump(X_list,open(location_X_list,'wb'))
            print('MXairfoil: X_failed loaded, trying to rescue')
        elif model == 'test':
            location_X_list = self.location+'/X_test_list.pkl'
            pickle.dump(X_list,open(location_X_list,'wb'))
            print('MXairfoil: X_test loaded, trying to rescue')
        elif model == 'testfun':
            self.model = 'testfun'
            location_X_list = self.location+'/X_testfun_list.pkl'
            pickle.dump(X_list,open(location_X_list,'wb'))
            print('MXairfoil: X_testfun loaded, trying to rescue') 
        elif model == 'test_rep':
            self.model = 'test_rep'
            location_X_list = self.location+'/X_test_rep_list.pkl'
            pickle.dump(X_list,open(location_X_list,'wb'))
            print('MXairfoil: X_test_rep loaded, trying to rescue')    
        elif model == 'demo180':
            # just calculate demo180
            y = self.calculate_demo180(X)
            self.jindu
            return y
        elif model == 're_calculate':
            # recalculate and update database.
            # location_X_list = self.location+'/X_list_re.pkl'
            # pickle.dump(X_list,open(location_X_list,'wb'))
            index_re_calculate = kargs['index']
            # filter X list, to be calculate.
            X_list_filter = [] 
            for i in range(len(X_list)):
                if X_list[i][0][0] == index_re_calculate:
                    X_list_filter.append(X_list[i])
            X_list=X_list_filter
            # then set jindu.
            self.jindu.N_points = 1
            self.jindu.N_calculated=0
        elif model == 'result_process':
            print('MXairfoil: running get_y_3D_mul2 for result_process')

        # first divide X into parts, and feed into threads.
        chicun = len(X_list)
        N_part = int(chicun / N) # max(int(chicun / N),1 )# at least 1 

        
        yushu = chicun - N_part*N # it is 0 if perfect.
        index_begin = 0 
        index_end = 0 
        for i in nloops:
            if i < yushu : 
                index_end = index_end + N_part+1
                # X_list_part = X_list[N_part*i:N_part*(i+1)+1]
            else:
                index_end = index_end + N_part
                # X_list_part = X_list[N_part*i:N_part*(i+1)]
            X_list_part = X_list[index_begin:index_end]
            index_begin = index_end + 0
            t = threading.Thread(target=self.get_result_3D_list,args=(i,X_list_part))
            threads.append(t)            
        
        #start the threads 
        for i in nloops:
            # time.sleep(i*3) #avoid same file name.
            # wochao yanzhong zuowu 
            time.sleep(3) #it is no longer necessary 
            threads[i].start()
        # waiting for the end of all threads
        for i in nloops:
            threads[i].join()

        #then get the results.
        for i in nloops:
            # y_part = pickle.load(open(self.location+'/'+str(i)+'.pkl','rb'))
            try:
                y_list_part = pickle.load(open(self.location+'/'+str(i)+'.pkl','rb'))
                if i==0 :
                    y_list = y_list_part
                else:
                    y_list = np.append(y_list,y_list_part,axis=0)    
            except:
                print('MXairfoil: cannnot get the pkl of thread '+str(i)+',  warning')
                y_list = np.array([])
        #get the y and save the y
        if model == 'train':
            location_y_list = self.location+'/y_list.pkl'
            pickle.dump(y_list,open(location_y_list,'wb'))
        elif model == 'test':
            location_y_list = self.location+'/y_test_list.pkl'
            pickle.dump(y_list,open(location_y_list,'wb'))
        elif model == 're_calculate':
            print('MXairfoil: re_calculate finished for one point')
            return 0 
        elif model == 'result_process':
            location_y_list = self.location+'/y_list_result_process.pkl'
            pickle.dump(y_list,open(location_y_list,'wb'))
            print('MXairfoil: finished get_y_3D_mul2 for result_process')

        # then post process, from X_list and y_list to x and y
        y = self.back_from_list(X_list,y_list)

        return y
    
    def calculate_demo180(self,X):
        # just calculate demo180, then return y.
        self.set_demo180()
        # self.diaoyong.point_max = np.array([0.3,0.7])
        # self.diaoyong.point_max = np.array([1.0,0.3])
        # point_max = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
        point_max = np.ones((self.x_dim,)) * 0.5 
        self.diaoyong.set_point_max(point_max)
        zhi_all = np.zeros((self.N_points,self.N_model))
        for i in range(self.N_points):
            zhi = self.diaoyong.ToulanFunction_general(X[i])
            zhi_all[i,:] = zhi
        self.jindu.N_calculated = self.jindu.N_points # set jindu to 100%
        return zhi_all

    def re_calculate_geo(self,index=[232],N_thread = 72):
        # load exsting X, then re calculate one of geometry point, then update the database (so-called database.)
        self.N_points = len(index)
        self.jindu = record_progress(N_points = self.N_points*self.N_steps_per_point)
        self.model='re_calculate'

        # load X 
        location_X = self.location+'/X.pkl'
        self.X = pickle.load(open(location_X,'rb'))

        # calculate
        for i in range(self.N_points):
            y = self.get_y_3D_mul2(self.X,N_thread,model=self.model,index=index[i])

        print('MXairfoil: finish re_calculate, En Taro XXH! ')

    def get_X_list(self,X):
        # this is to generate am X_list for further simulation.
        X_list = []
        N_geometry = len(X)

        X_zhongjie = []
        for i_geometry in range(N_geometry):
            for i_CFD in range(self.N_steps_per_point):
                zhuansu = self.zhuansu 
                # yali = self.Pout_min + self.Pout_d * i_CFD 
                yali = self.Pout_min + self.Pout_dianlie[i_CFD] 
                X_CFD = [zhuansu,yali] 
                X_zhongjie.append([i_geometry,i_CFD])
                X_zhongjie.append(X[i_geometry])
                X_zhongjie.append(X_CFD)
                X_list.append(X_zhongjie)
                X_zhongjie = []

        return X_list

    def sort_X_list(self,X_list,index_geometry,index_CFD):
        # find X by indeies from list.
        N_list = len(X_list)

        # very baoli, it could be accelerate.
        for i in range(N_list):
            if (X_list[i][0][0] == index_geometry)&(X_list[i][0][1] == index_CFD):
                return X_list[i]

    def back_from_list(self,X_list,y_list):
        # do some check.
        len_x = len(X_list)
        len_y = len(y_list)
        
        if (len_x != len_y) | (len_y != self.N_points * self.N_steps_per_point):
            raise Exception('MXairfoil: wrong CFD result, G!')

        # then post process, from X_list and y_list to x and y
        from call_components import curve 

        N_geometry = self.N_points
        y = []
        for i in range(N_geometry):
            if i == 232: #67 232
                print('MXairfoil: skimming data, now number ' + str(i) + ' geometry points.')
                pass # this is for debug
            y_curve = np.array([]).reshape(0,4)
            for i_search in range(len_x):
                if X_list[i_search][0][0] == i:
                    if len(y_curve) == 0 :
                        y_curve =y_list[i_search].reshape(1,4)
                    else:
                        y_curve = np.append(y_curve,y_list[i_search].reshape(1,4),axis=0)

            # a little check.
            if y_curve[-1][0]>y_curve[0][0]:
                # which means there are more flow rate when P_out increased
                raise Exception('MXairfoil: it seems CFD wrong.')
            
            y_curve = self.check_y_curve(y_curve)
            # calculate criterias 
            massflow_ave = y_curve[:,0] 
            massflow_rate = massflow_ave / self.massflow_0 
            # 3, fit these data into a curve.       
            data_efficiency = np.array([massflow_rate ,y_curve[:,2]])
            data_efficiency = data_efficiency.T
            curve_efficiency = curve(data_efficiency,x_name='massflow_rate',y_name='efficiency',title='efficiency')
            data_pi = np.array([massflow_rate,y_curve[:,1]])
            data_pi = data_pi.T
            curve_pi = curve(data_pi,x_name='massflow_rate',y_name='pi',title='pi'
            )

            # 3.5, get so-called working point. the max efficiency point selected.
            eta_w = max(y_curve[:,2])
            zhi = curve_efficiency.y_to_x(eta_w)
            # massflow_rate_w = 0.5*(zhi[0]+zhi[1])
            massflow_rate_w = zhi.mean()
            pi_w = curve_pi.x_to_y(massflow_rate_w)

            # 4, integration and give result.
            # efficiency_threshold = 0.87 # 0.88 might be too high. 
            zhi = curve_efficiency.y_to_x(self.efficiency_threshold) # there should be two x
            zhi = zhi[~np.isnan(zhi)]
            try:
                massflow_rate_lower = zhi.min() 
                massflow_rate_upper = zhi.max()
            except:
                # if no area to integral
                massflow_rate_lower = min(data_efficiency[0][:])
                massflow_rate_upper = max(data_efficiency[0][:])
                efficiency_integration = 0 
                pi_integration = 0 
                # return massflow_rate_lower,massflow_rate_upper,efficiency_integration,pi_integration,massflow_rate_w,eta_w,pi_w 

            efficiency_integration = curve_efficiency.integral(massflow_rate_lower,massflow_rate_upper)
            pi_integration = curve_pi.integral(massflow_rate_lower,massflow_rate_upper)

            efficiency_panju = efficiency_integration/(massflow_rate_upper - massflow_rate_lower)
            pi_panju = pi_integration / (massflow_rate_upper - massflow_rate_lower)

            # return massflow_rate_lower,massflow_rate_upper,efficiency_integration,pi_integration,massflow_rate_w,eta_w,pi_w # 7 parameters.
            y_geometry_single = [massflow_rate_lower,massflow_rate_upper,efficiency_panju,pi_panju,massflow_rate_w,eta_w,pi_w] 

            y_geometry_single_limit = self.limit_y(y_geometry_single)
            y.append(y_geometry_single_limit) # 7 parameters.
            print('MXairfoil: finish processing nomber '+ str(i) + ' geometry point')
        y = np.array(y)
        print('MXairfoil: N_limited=' + str(self.N_limited))
        return y 

    def debug_back_from_list(self,N_points=300,**kargs):
        # this is to debug self.back_from_list.
        # read X_list, y_list and call self.back_from_list.
        self.N_points = N_points
        self.N_steps_per_point = 40 
        self.N_fail = 0 
        self.jindu = record_progress(N_points = self.N_points*self.N_steps_per_point)
        
        if 'model' in kargs:
            model = kargs['model']
        else:
            model = 'train'
        
        location_X_list = self.location+'/X_list.pkl'
        X_list = pickle.load(open(location_X_list,'rb'))

        if model == 'rescue':
            # get y_list from storage.
            
            diaoyong = call_components(self.script_folder,self.matlab_location,case=self.case)
            geshu = len(X_list)
            zhi = np.zeros([geshu,4])
            for index_N in range(geshu):
                X = X_list[index_N]
                flag,location = self.jindu.check_3D_result(index_geometry = X[0][0], index_CFD = X[0][1])
                if flag == True:
                    # which means this has been calculated
                    zhi[index_N][0],zhi[index_N][1],zhi[index_N][2] = diaoyong.get_result_3D(resultLocation=location)
                else: 
                    self.jindu.GG_1ci(X)
                    zhi[index_N][0] = 0 
                    zhi[index_N][1] = 0 
                    zhi[index_N][2] = 0 
                    print('MXairfoil: one X is not find, G')
            y_list = zhi
        elif model == 'train':
            location_X_list = self.location+'/X_list.pkl'
            location_y_list = self.location+'/y_list.pkl'

            X_list = pickle.load(open(location_X_list,'rb'))
            y_list = pickle.load(open(location_y_list,'rb'))

        y = self.back_from_list(X_list,y_list)

        #get the y and save the y
        location_y = self.location+'/y_CFD.pkl'
        pickle.dump(y,open(location_y,'wb'))

        print('MXairfoil: GG points: ' + str(self.N_fail)) 

        return y 

    def check_y_curve(self,y_curve):
        # check and modify y_curve in back_from_list.
        if y_curve[-1][0]>y_curve[0][0]:
            # which means there are more flow rate when P_out increased
            raise Exception('MXairfoil: it seems CFD wrong.')

        # zuobi 
        y_curve = y_curve[np.argsort(y_curve[:,0]),:]
        y_curve = np.flipud(y_curve)
        # zuobi 

        geshu = len(y_curve)
        # self.massflow_threshold = 20 
        massflow_threshold = self.massflow_threshold
        for i in range(geshu):
            # eliminate unexpected [0,nan,nan] CFD point
            if (y_curve[i,0] < massflow_threshold*0.1 ) and (np.isnan(y_curve[i,1])):
                if (i >= 1) and (i <= geshu-1):
                    # then average.
                    y_curve[i] = (y_curve[i-1] + y_curve[i+1])/2.0
                elif i == 0 :
                    y_curve[i] = y_curve[i+1]
                elif i == geshu-1:
                    y_curve[i] = y_curve[i-1]

            # eliminate useless CFD point
            if y_curve[i,0]< massflow_threshold:
                if max(y_curve[i:,0]) > massflow_threshold*1.3:
                    raise Exception('MXairfoil: invalid y_curve, G!')
                index_threshold = i
                massflow_compensate = y_curve[i,0]
                eta_compensate = 0 # y_curve[i-1,1]
                pi_compensate = 0 # y_curve[i-1,2]
                break
            else:
                index_threshold = i
                massflow_compensate = y_curve[i,0]
                eta_compensate = y_curve[i,2]
                pi_compensate = y_curve[i,1]     
        
        y_curve[index_threshold:,0] = massflow_compensate
        y_curve[index_threshold:,1] = pi_compensate
        y_curve[index_threshold:,2] =  eta_compensate
        self.N_fail = self.N_fail + geshu - index_threshold

        # if it never stall
        if (y_curve[geshu-1,0]>massflow_threshold) and (y_curve[geshu-1,2]>self.efficiency_threshold):
            y_curve[geshu-1,1]= y_curve[geshu-1,1]*0.9
            y_curve[geshu-1,2]= y_curve[geshu-1,2]*0.9
            # to ensure that intergation right.
            print('MXairfoil: attension, one point seems never stall.')


        return y_curve

    def limit_y(self,y):
        # give a limit for each dimension of y, in back_from_list
        for i in range(len(y)):
            # limit the value.
            if y[i]<self.y_limit_l[i]:
                chazhi = self.y_limit_l[i] - y[i] 
                fanwei = (self.y_limit_u[i]-self.y_limit_l[i])
                chazhi_new = self.xianzhi(chazhi,fanwei)
                y[i] = self.y_limit_l[i] - chazhi_new
            elif y[i]>self.y_limit_u[i]:
                chazhi = y[i] - self.y_limit_u[i]
                fanwei = (self.y_limit_u[i]-self.y_limit_l[i])
                chazhi_new = self.xianzhi(chazhi,fanwei)
                y[i] = self.y_limit_u[i] + chazhi_new
            else:
                # nothing happen.
                pass                    

        return y                
    def xianzhi(self,chazhi,fanwei):
        xishu = self.xishu_limit# 3.0 # indicate how strong the limitation is.
        if xishu<0.000001:
            return chazhi
        chazhi_normal = xishu*chazhi/fanwei
        chazhi_normal_new = np.log(chazhi_normal + 1.0 )
        chazhi_new = chazhi_normal_new/xishu*fanwei
        self.N_limited = self.N_limited + 1
        return chazhi_new

    def get_Pout_distribution(self,Prange,**kargs):
        # self.Pout_min = 128591.6 - 14000, Prange = 14000*2
        N = self.N_steps_per_point
        if 'model' in kargs:
            model = kargs['model']
        else:
            model = 'dengbi'
        if model == 'dengbi':
            # Pout_d_last = 80 
            # Pout_d_first = 1000 
            # q =  (Pout_d_first / Pout_d_last)**(1/N)
            if N==40 :
                q = 0.92 # this is for N = 40
            elif N==20:
                q = 0.9
            a1 = (1-q)/(1-q**N)
            dianlie = [a1*(1-q**i)/(1-q)*Prange for i in range(N)]
        elif model == 'cifang':
            cifangshu = 1.7
            dianlie = [(1-((N-i)/N)**cifangshu)*Prange for i in range(N)]


        self.Pout_dianlie = dianlie

        return  dianlie # this is for debug.

    def train_model_mul_3D(self,N,N_thread,add_original = False,xishu_limit=3,**kargs):
        self.N_points = N
        self.N_steps_per_point = 40 
        dim = self.N_model
        self.pmin = kargs['pmin']
        self.thetamax = kargs['thetamax']
        self.xishu_limit = xishu_limit
        if (len(self.pmin)!=dim) or (len(self.pmin)!=dim):
            raise Exception('MXairfoil: invalid kriging model setting')


        if 'model' in kargs:
            model_X_rescue = kargs['model']
            print('MXairfoil: model selected in train_model_mul_3D, ' + model_X_rescue)
        else:
            model_X_rescue = 'train'

        location_X = self.location+'/X.pkl'
        try:
            self.X = pickle.load(open(location_X,'rb'))
        except:
            self.X = self.sp.optimallhc(N)
            pickle.dump(self.X,open(location_X,'wb'))

        if len(self.X) != self.N_points:
            print('MXairfoil: forced use first ' + str(self.N_points) + ' trainning points')
            self.X = self.X[0:self.N_points]
        elif len(self.X) < self.N_points:
            raise Exception('MXairfoil: G. invalid trainning input')

        if add_original:
            self.X = self.add_original_point(self.X)
            self.N_points = self.N_points+1 
            pickle.dump(self.X,open(location_X,'wb'))
        
        print('MXairfoil: X generated.')
        self.jindu = record_progress(N_points = self.N_points*self.N_steps_per_point)
        # massflow_rate_lower,massflow_rate_upper,efficiency_integration,pi_integration
        
        testfun1 = self.testfun_massflow_rate_lower
        testfun2 = self.testfun_massflow_rate_upper
        testfun3 = self.testfun_efficiency_integration
        testfun4 = self.testfun_pi_integration
        testfun5 = self.testfun_massflow_rate_w
        testfun6 = self.testfun_eta_w
        testfun7 = self.testfun_pi_w
        # massflow_rate_lower,massflow_rate_upper,efficiency_integration,pi_integration,massflow_rate_w,eta_w,pi_w
        self.testfunx_list = [testfun1,testfun2,testfun3,testfun4,testfun5,testfun6,testfun7]  
        if len(self.testfunx_list) != self.N_model:
            raise Exception('MXairfoil: invalid testfun, G! ')
        if model_X_rescue == 'demo180':
            self.N_model=2 
            dim = 2

        #get the y and save the y
        location_y = self.location+'/y_CFD.pkl'
        try:
            y = pickle.load(open(location_y,'rb'))
        except:
            # X_real = self.bianhuan.surrogate_to_real(self.X)
            y = self.get_y_3D_mul2(self.X,N_thread,model=model_X_rescue)
            while self.jindu.check_process_CFD() < 0.99:
                self.model = 'rescue'
                y = self.get_y_3D_mul2(self.X,N_thread,model=self.model)
                # raise Exception('MXairfoil: CFD have not finished, G!')
            self.model=model_X_rescue
            pickle.dump(y,open(location_y,'wb'))
        
        try:
            y=np.array(y)
            y = y.astype('float64')
        except:
            pass
        if type(self.X) == type(y):
            if self.X.dtype == y.dtype:
                pass
            else:
                raise Exception('MXairfoil: invalid variable type, G!')
        else :
            raise Exception('MXairfoil: invalid variable type, G!')

        
        self.yx_list = []
        for i in range(dim):
            self.yx_list.append(y[:,i])

         
        # this should not even work if everything is right.
        # flag = 1
        # while flag >0 :
        #     flag = self.check_existing_data_3D()

        self.kx_list = []

        for i in range(dim):
            ki = kriging(self.X, self.yx_list[i], testfunction=self.testfunx_list[i], name='simple') 
            # ki.pmin= self.pmin[i]# 1.5
            # ki.thetamax=self.thetamax[i]# 10
            # ki.train()
            self.kx_list.append(ki)
        self.k_train_mul(self.kx_list)
        
        self.save()
        
        return self.yx_list
    def k_train_single(self,i):
        ki = self.kx_list[i]
        ki.pmin= self.pmin[i]# 1.5
        ki.thetamax=self.thetamax[i]# 10  
        ki.train()      

    def check_existing_data_3D(self, **kargs):
        y_list = self.yx_list
        X = self.X
        # this is to check the data before trainning. to avoid wasting time.
        N_y = len(y_list)
        y_lower = [0.5,0.6,0.01,0.02,0.8,0.8,1.5]
        y_upper = [1.2,1.3,0.2,0.3,1.1,1,1.8]
        y_gg_index = [] 
        panju = 0 
        for i in range(N_y):
            yi_gg_index =np.argwhere(y_list[i]<y_lower[i]) # this is (n,1)
            yi_gg_index = np.append(yi_gg_index,np.argwhere(y_list[i]>y_upper[i]),axis=0)
            panju = panju + len(yi_gg_index)
            y_gg_index.append(yi_gg_index)
        
        gg_index = np.append(y_gg_index[0],y_gg_index[1],axis=0)
        for i in range(N_y-2):
            gg_index = np.append(y_gg_index[i+2],gg_index,axis=0)
        gg_index = np.unique(gg_index)
        gg_index = gg_index.reshape(len(gg_index),)
        X_gg = X[gg_index.astype(int)]

        return len(X_gg)

    def test_repeatability_3D(self):
        # this is to test the repeatability of CFD parts.
        
        location_X = self.location+'/X.pkl'
        self.X = pickle.load(open(location_X,'rb'))
        print('MXairfoil: X loaded.')
        x_test = self.X[0].reshape(1,self.x_dim)
        self.N_points = 4
        self.jindu = record_progress(N_points = self.N_points*self.N_steps_per_point)

        X_rep = np.append(x_test,x_test,axis=0)
        X_rep = np.append(X_rep,X_rep,axis=0)
        
        print('MXairfoil: X generated for testing repeatability.')

        #get the y and save the y
        location_y = self.location+'/y_rep.pkl'
        try:
            y = pickle.load(open(location_y,'rb'))
        except:
            # X_real = self.bianhuan.surrogate_to_real(self.X)
            y = self.get_y_3D_mul2(X_rep,72,model='test_rep')
            pickle.dump(y,open(location_y,'wb'))
            
        y=np.array(y)
        y = y.astype('float64')

        for i in range(7):
            norml_err = np.sqrt(np.var(y[:,i])) /np.mean(y[:,i])
            print('MXairfoil: this is dimension '+str(i) + ', the variance is ' + str(norml_err))  
        
        return 0 

    def recycle_result_3D(self,N_thread):
        # this is to recycle something after a process GGed
        diaoyong =  call_components(self.script_folder,self.matlab_location,case=self.case)
        for i in range(N_thread):
            diaoyong.test_existing_case(i+1,case=self.case)
            massflow_rate_lower,massflow_rate_upper,efficiency_integration,pi_integration=diaoyong.result_process_3D()
            diaoyong.reset_result_3D(collect=True,clear=False) # clear and collect result.
        diaoyong.clear_all(self.script_folder,self.matlab_location)

    def test_to_train(self,flag_force=True):
        # transfer existing testing data into trainning data, then delet X_test and X_test list.
        # self.load()
        
        # load X and X_test 
        location_X = self.location+'/X.pkl'
        location_y = self.location+'/y_CFD.pkl'
        X_train = pickle.load(open(location_X,'rb'))
        location_X_test = self.location+'/X_test.pkl'
        location_y_test = self.location+'/y_test_CFD.pkl'
        location_y_list = self.location+'/y_list.pkl'
        location_y_test_list = self.location+'/y_test_list.pkl'        
        if os.path.exists(location_y_test) or flag_force:
            pass 
        else:
            print('MXairfoil: it seems nothing to transfer')
            return len(X_train)
        X_test = pickle.load(open(location_X_test,'rb'))

        X_train_new= np.append(X_train,X_test,axis=0)
        self.X = X_train_new
        pickle.dump(X_train_new,open(location_X,'wb'))


        # then manage database
        X_train_list = self.get_X_list(X_train)    
        X_test_list = self.get_X_list(X_test)
        X_list = self.get_X_list(X_train_new)
        self.jindu = record_progress(N_points = len(X_train_list) + len(X_test_list))
        # X_list = self.jindu.check_3D_X_list(X_list,model=model)
        # wanchengdu = self.jindu.check_process_CFD()           
        self.jindu.test_to_train(X_test_list,X_train_list)
        
        os.remove(location_y)
        os.remove(location_y_test)
        os.remove(location_y_list)
        os.remove(location_y_test_list)
        # os.remove(location_X_test)
        for i in range(len(self.yx_list)):
            location_yy_i = self.location+'/yy'+ str(i+1) +'.pkl'  
            location_y_i = self.location+'/y'+ str(i+1) +'.pkl'  
            location_k_i = self.location+'/k'+ str(i+1) +'.pkl'  
            os.remove(location_yy_i)
            os.remove(location_y_i)
            os.remove(location_k_i)
        return len(X_train_new)

    def train_to_test(self,index_begin=300,index_end=315):
        # load X and X_test 
        location_X = self.location+'/X.pkl'
        location_y = self.location+'/y_CFD.pkl'
        location_X_test = self.location+'/X_test.pkl'
        location_y_test = self.location+'/y_test_CFD.pkl'
        location_y_list = self.location+'/y_list.pkl'
        location_y_test_list = self.location+'/y_test_list.pkl' 

        # 1, check 
        X_test = pickle.load(open(location_X_test,'rb'))       
        X_train = pickle.load(open(location_X,'rb'))       
        if os.path.exists(location_X_test):
            print('MXairfoil: existing X_test')
            # return len(X_test)
        else:
            print('MXairfoil: no X_test, get one')

        # 2, devide X 
        X_test_new = X_train[index_begin:index_end]
        X_train_new_qian = X_train[0:index_begin]
        X_train_new_hou = X_train[index_end:]
        
        X_train_new = np.append(X_train_new_qian,X_train_new_hou,axis=0)

        X_train_list = self.get_X_list(X_train) 
        X_test_new_list = self.get_X_list(X_test_new) 
        X_train_new_list = self.get_X_list(X_train_new)

        X_train_qian_list = X_train_list[0:index_begin*self.N_steps_per_point]
        X_train_new_qian_list = X_train_new_list[0:index_begin*self.N_steps_per_point] # X_train_qian_list should equal to X_train_new_qian_list
        
        X_test_list = X_train_list[index_begin*self.N_steps_per_point:index_end*self.N_steps_per_point]

        X_train_hou_list = X_train_list[index_end*self.N_steps_per_point:]
        X_train_new_hou_list = X_train_new_list[index_begin*self.N_steps_per_point:]
        # 3, rename as testcase
        self.jindu = record_progress(N_points = len(X_train_new_qian_list) + len(X_train_new_hou_list))

        self.jindu.train_to_test(X_test_list,X_test_new_list,model_new='test')

        # 4, rename other case
        self.jindu.train_to_test(X_train_hou_list,X_train_new_hou_list,model_new='train')

        # 5, save X and X_test. remove y
        self.X = X_train_new
        pickle.dump(X_train_new,open(location_X,'wb'))
        pickle.dump(X_test_new,open(location_X_test,'wb'))

        os.remove(location_y)
        os.remove(location_y_test)
        os.remove(location_y_list)
        os.remove(location_y_test_list)
        # os.remove(location_X_test)
        for i in range(len(self.yx_list)):
            location_yy_i = self.location+'/yy'+ str(i+1) +'.pkl'  
            location_y_i = self.location+'/y'+ str(i+1) +'.pkl'  
            location_k_i = self.location+'/k'+ str(i+1) +'.pkl'  
            os.remove(location_yy_i)
            os.remove(location_y_i)
            os.remove(location_k_i)
        return len(X_train_new)        
        pass 

    def k_train_mul(self,k_list):
        print('MXairfoil: start trainning kriging model in parallel')
        N_thread = len(k_list)
        threads = []  

        for i in range(N_thread):
            t = threading.Thread(target=self.k_train_single,args=(i,))
            threads.append(t)
        #start the threads 
        for i in range(N_thread):
            # time.sleep(i*3) #avoid same file name.
            # wochao yanzhong zuowu 
            time.sleep(3) #it is no longer necessary 
            threads[i].start()
        # waiting for the end of all threads
        for i in range(N_thread):
            threads[i].join()
        
        print('MXairfoil: finish trainning kriging model in parallel')

    def add_original_point(self,X):
        # add original point and translate as a case.
        X_original = self.bianhuan.real_to_surrogate(self.state_original_real)
        X_original = X_original.reshape(1,self.x_dim)
        X_new = np.append(X,X_original,axis=0)
        return X_new
        pass 

    def debug_Rotor_case(self,model,**kargs):
        # just test the things, trying to find where the wrong is.
        if model == 'CFD':
            # only test the CFD case, target location is needed.
            location = kargs['location']
            diaoyong = call_components(self.script_folder,self.matlab_location,case=self.case,source_script_folder=location) 
            diaoyong.call_matlab()
            diaoyong.call_AutoGrid5()
            # diaoyong.call_Turbo()
        elif model == 'All':
            # test All, X_list and index are needed
            location_X_list = kargs['location']
            index_geometry = kargs['index_geometry']
            index_CFD = kargs['index_CFD']
            X_list = pickle.load(open(location_X_list,'rb'))
            X_in = self.sort_X_list(X_list,index_geometry,index_CFD)
            diaoyong = call_components(self.script_folder,self.matlab_location,case=self.case) 
            zhi = [0,0,0]
            zhi[0],zhi[1],zhi[2] = self.jisuan_3D_mul2(diaoyong,X_in) 
        elif model == 'step':
            # test matlab part, X_list and index are needed
            location_X_list = kargs['location']
            index_geometry = kargs['index_geometry']
            index_CFD = kargs['index_CFD']
            location_X_list = kargs['location']
            index_geometry = kargs['index_geometry']
            index_CFD = kargs['index_CFD']
            X_list = pickle.load(open(location_X_list,'rb'))
            X_in = self.sort_X_list(X_list,index_geometry,index_CFD)
            diaoyong = call_components(self.script_folder,self.matlab_location,case=self.case) 
            zhi = [0,0,0]
            # 1,build the geometry.
            X_real = self.bianhuan.surrogate_to_real(X_in[1])
            diaoyong.set_value_3D(X_real)
            diaoyong.reset_result_3D() # clean and do not collect
            diaoyong.call_matlab()
            diaoyong.call_AutoGrid5()
    def testfun_kernel(self,x):
        # read x_test and judge if they are equal.
        location_X_test  = self.location+'/X_testfun.pkl'
        location_Y_test  = self.location+'/Y_testfun.pkl'
        try:
            x_exist = pickle.load(open(location_X_test,'rb'))
            y_exist = pickle.load(open(location_Y_test,'rb'))
        except:
            # x_exist = x 
            # pickle.dump(x,open(location_X_test,'wb'))
            print('MXairfoil: test function. there is no existing data.')
        
        if ~isinstance(x,list):
            print('MXairfoil: invaild input. trying to rescue')
            x = [x]
        changdu_exist = len(x_exist)
        changdu = len(x)
        y = list(range(changdu))
        x_not_exist = [] 
        index_not_exist = [] 
        index_exist = [] 
        for i in range(changdu): 
            # for i_exist in range(changdu_exist):
            #     if x[i] == x_exist[i_exist]:
            if x_exist.count(x[i])>0 :
                # this x has been calculated
                index = x_exist.index(x[i])
                y[index] = y_exist[index]
                index_exist.append(i)
            else:
                # this x has not been calculated.
                x_not_exist.append(x[i])
                index_not_exist.append(i)
        
        # check 
        if len(y)!=(len(index_exist)+len(index_not_exist)):
            raise Exception('MXairfoil: test function wrong, G')

        # then calculate x_not_exist
        self.model = 'testfun'
        y_not_exist = self.get_y_3D_mul2(x_not_exist,64,model='testfun') 
        for i in range(len(index_not_exist)):
            y[index_not_exist[i]] = y_not_exist[i]
        # then all y should have been calculated.
        # then save updated and y 
        x_exist.extend(x_not_exist)
        y_exist.extend(y_not_exist)
        pickle.dump(x_exist,open(location_X_test,'wb'))
        pickle.dump(y_exist,open(location_Y_test,'wb'))

        return y 
    def testfun_massflow_rate_lower(self,x):
        # even there is only one x, there are self.N_steps_per_point CFD points, so multiprocess is always need
        zhi = self.testfun_kernel(x)
        return zhi[:,0]
    def testfun_massflow_rate_upper(self,x):
        # this might cost too much time, but I have no choice so far.
        zhi = self.testfun_kernel(x)
        return zhi[:,1]
    def testfun_efficiency_integration(self,x):
        # this might cost too much time, but I have no choice so far.
        zhi = self.testfun_kernel(x)
        return zhi[:,2]
    def testfun_pi_integration(self,x):
        # this might cost too much time, but I have no choice so far.
        zhi = self.testfun_kernel(x)
        return zhi[:,3]
    def testfun_massflow_rate_w(self,x):
        zhi = self.testfun_kernel(x)
        return zhi[:,4]
    def testfun_eta_w(self,x):
        zhi = self.testfun_kernel(x)
        return zhi[:,5]
    def testfun_pi_w(self,x):
        zhi = self.testfun_kernel(x)
        return zhi[:,6]
    # massflow_rate_w,eta_w,pi_w

    def set_demo180(self):
        if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
            sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/main')
        elif os.environ['COMPUTERNAME'] == 'DESKTOP-132CR84' :
            sys.path.append(r'D:/XXHcode/DDPGshishi/main')
        
        from DemoFunctions import DemoFunctions
        self.diaoyong = DemoFunctions()
        self.N_model=2

class record_progress:
    def __init__(self,**kargs) :
        self.N_points = kargs['N_points']
        self.N_calculated = 0 
        self.N_failed = 0 
        self.X_failed = []
        self.flag_Xgenerate = 0
        self.flag_Ktrain = 0 
        self.flag_Datacheck = 0   
        if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
            #which means in my diannao
            self.HDD_location = 'E:/EnglishMulu'
        elif os.environ['COMPUTERNAME'] == 'DESKTOP-132CR84' :
            # which means in new working zhan. D:\XXHdatas\EnglishMulu
            # raise Exception('MXairfoil: no location setted, G')
            self.HDD_location = 'E:/XXHdatas/EnglishMulu'
        else:
            # which means in 106 server   
            self.HDD_location = 'C:/Users/106/Desktop/EnglishMulu'
        self.location_X_failed = self.HDD_location + '/X_faided.pkl'
        pass
    def check_progress(self):
        # this is to check the process while running.
        rizhi = 'MXairfoil: check the process... \n    generating X:' + str(self.flag_Xgenerate) + '\n    calculating CFD: ' + str(round(100.0*self.N_calculated/self.N_points,2)) + '% \n    data checking: ' + str(self.flag_Datacheck)+ '\n    generating kriging model:'+ str(self.flag_Xgenerate) + '\n    N_failed: '+ str(self.N_failed)
        print(rizhi)
        
        # if 'self.flag_Xgenerate' in vars():
        # if hasattr(self,'flag_Xgenerate'):
        #     rizhi = 'MXairfoil: check the process... \n    generating X:' + str(self.flag_Xgenerate) + '\n    calculating CFD: ' + str(100.0*self.N_calculated/self.N_points) + '% \n    generating kriging model:'+ str(self.flag_Xgenerate)
        #     print(rizhi)
        #     # if calling it every calculating
        #     self.N_calculated = self.N_calculated+1
        # else:
        #     # which means calling it for the first time.
        #     self.N_calculated = 0 
        #     self.flag_Xgenerate = 0
        #     self.flag_Ktrain = 0 
        #     self.flag_datacheck = 0 
        return rizhi
    def check_process_CFD(self):
        jindu = self.N_calculated/self.N_points*1.0
        return jindu 
    def calculate_1ci(self,**kargs):
        self.N_calculated = self.N_calculated + 1 
        if 'tips' in kargs:
            tips = kargs['tips']
            print('MXairfoil: calculated_1ci, '+tips)
        return self.N_calculated
    def Xgenerate_done(self):
        self.flag_Xgenerate = 1
    def Krigingtrain_done(self):
        self.flag_Ktrain = 1
    def Detacheck_done(self):
        self.flag_Datacheck = 1         
    def paoyixia(self):
        while(1):
            time.sleep(1)
            self.check_progress()
            self.N_calculated = self.N_calculated +1 
    
    def save_3D_process(self,weizhi_folder):
        # this is to save process.
        weizhi = weizhi_folder + 'record_progress.pkl'
        pickle.dump(self,open(weizhi,'wb'))
        return 0
    def load_3D_process(self,weizhi_folder):
        # this is to load the process
        weizhi = weizhi_folder + 'record_progress.pkl'
        self = pickle.load(open(weizhi,'rb'))
        return self

    def save_3D_result(self,location_temp,**kargs):
        # this is to copy cases into HDD
        index_geometry = kargs['index_geometry']
        index_CFD = kargs['index_CFD'] 
        qianzhui = self.get_qianzhui(kargs)
        location_target = self.HDD_location + qianzhui+str(index_geometry)+'CFD'+str(index_CFD)
        try:
            if kargs['model'] == 're_calculate':
                print('MXairfoil: re_calculate and update one CFD points')
                shutil.rmtree(location_target)
            shutil.copytree(location_temp,location_target)
        except:
            print('MXairfoil: warning! repetitive case?')
            i=1
            location_target2 = location_target + '('+str(i)+')'
            while (i<1000)&(os.path.exists(location_target2)):
                location_target2 = location_target + '('+str(i)+')'
                shutil.copytree(location_temp,location_target2)

    def get_qianzhui(self,**kargs):
        if 'model' in kargs:
            if kargs['model']== 'test':
                qianzhui = '/testcase'
            elif kargs['model'] == 'testfun':
                qianzhui = '/testfuncase'
            elif kargs['model'] == 'test_rep':
                qianzhui = '/test_repcase'
            elif kargs['model'] == 're_calculate':
                qianzhui = '/case'
            elif kargs['model'] == 'result_process':
                qianzhui = '/resultcase'
            else:
                qianzhui = '/case_undefined'
        else:
            qianzhui = '/case'

        return qianzhui

    def check_3D_result(self,**kargs):
        # if this point have been calculated, then reture True, else reture False
        index_geometry = kargs['index_geometry']
        index_CFD = kargs['index_CFD'] 
        qianzhui = self.get_qianzhui(kargs)
        location_target = self.HDD_location + qianzhui+str(index_geometry)+'CFD'+str(index_CFD)   
        check_result = os.path.exists(location_target)
        return check_result,location_target

    def GG_1ci(self,X):
        # save the failed X
        self.X_failed.append(X)
        self.N_failed = self.N_failed + 1
        print('MXairfoil: one X faild, total : '+ str(self.N_failed))
        pickle.dump(self.X_failed,open(self.location_X_failed,'wb'))

    def check_3D_X_list(self,X_list,model = 'train'):
        # check and return Xs that have not been calculated.
        X_list_new = [] 
        N_X = len(X_list)
        N_finished = 0 
        for i in range(N_X):
            flag,location = self.check_3D_result(index_geometry = X_list[i][0][0], index_CFD = X_list[i][0][1],model=model)
            if flag == True:
                pass
            else :
                X_list_new.append(X_list[i])
        N_finished = N_X - len(X_list_new)
        self.N_calculated = self.N_calculated + N_finished
        print('MXairfoil: check X_list, ' + str(N_finished) + ' finished. ')
        return X_list_new

    def test_to_train(self,X_test_list,X_train_list):
        # one by one, copy.
        index_geometry_old_max = X_train_list[-1][0][0]
        for i in range(len(X_test_list)):
            index_geometry_new = X_test_list[i][0][0] + index_geometry_old_max + 1
            flag_new,location_new = self.check_3D_result(index_geometry = index_geometry_new, index_CFD = X_test_list[i][0][1],model='train')
            flag_old,location_old = self.check_3D_result(index_geometry = X_test_list[i][0][0], index_CFD = X_test_list[i][0][1],model='test')
            if (flag_new == False) and (flag_old == True):
                # shutil.copytree(location_old,location_new)
                # shutil.rmtree(location_old)
                os.rename(location_old,location_new)
                print('MXairfoil: transfered one CFD point to trainning data, new geometry index=' + str(index_geometry_new))
                pass
            else:
                raise Exception('MXairfoil: Wrong when transfer test data into trainning data.')
    
    def train_to_test(self,X_train_list,X_test_list,model_new = 'test'):
        # part of the X_train_list is X_test_list, copy.
        # check
        if len(X_train_list) != len(X_test_list):
            raise Exception('MXairfoil: GG in train_to_test')
        for i in range(len(X_test_list)):
            flag_old,location_old = self.check_3D_result(index_geometry = X_train_list[i][0][0], index_CFD = X_train_list[i][0][1],model='train')
            flag_new,location_new = self.check_3D_result(index_geometry = X_test_list[i][0][0], index_CFD = X_test_list[i][0][1],model=model_new)
            if (flag_new == False) and (flag_old == True):
                # shutil.copytree(location_old,location_new)
                os.rename(location_old,location_new)
                print('MXairfoil: transfered one CFD point to ' + model_new + ' data, new geometry index=' + str(X_test_list[i][0][0]))
                pass
            else:
                raise Exception('MXairfoil: Wrong when transfer trainning data into testing data.')            

if __name__ =='__main__':
    print('MXairfoil: test the Surrogate')
    total_time_start = time.time()
    # shishi = Surrugate(case='NACA65')
    # shishi = Surrugate(case='Rotor67',x_dim=18)
    flag = 7 # 1 for train, -1 for clear, 2 for test all. 5 for single point . 7 for 3D point, and trainning, 8 for debug.
    if flag ==0 :
        shishi = Surrugate(case='CDA1')
        shishi.load()
    elif flag == -1:
        shishi = Surrugate(case='CDA1')
        shishi.clear()
    elif flag == 1:
        shishi = Surrugate(case='CDA1')
        y1,y2,y3 = shishi.train_model_mul(3**4,9)
        # execute a simple test.
        y1_predict = shishi.k1.predict(shishi.X[0])
        y2_predict = shishi.k2.predict(shishi.X[0])
        y3_predict = shishi.k3.predict(shishi.X[0])
        print('MXairfoil: test the trained Surrogate','\n y1[0] = ' , y1[0] , '\n its predict = ' , y1_predict ,'\n y2[0] = ' , y2[0] , '\n its predict = ' , y2_predict, '\n y3[0] = ' , y3[0] , '\n its predict = ' , y3_predict,'\nN_correct=',shishi.N_correct)
        shishi.save()
        shishi.test_all(50)
        shishi.auto_record()
    elif flag == 2:
        shishi = Surrugate(case='CDA1')
        shishi.test_all(50)
        shishi.auto_record()
    elif flag == 3:
        shishi = Surrugate(case='CDA1')
        shishi.test_repeatability()
    elif flag ==4:
        shishi = Surrugate(case='CDA1')
        shishi.test_data()
    elif flag == 5:
        shishi = Surrugate(case='NACA65')
        # calculate a detected optimization point.
        original_NACA65_surrogate_state = np.array([0.46570833, 0.45239167, 0.557     , 0.50823   ])
        detected_optimization_surrogate = np.array([0.06831629,0.09575173])
        input_surrogate = original_NACA65_surrogate_state
        input_surrogate[0:2] = detected_optimization_surrogate
        from transfer import *
        bianhuan = transfer(tishi=1)
        bianhuan.real_obs_space_l = np.array([0.37,-0.34,0.045,0.35])
        bianhuan.real_obs_space_h = np.array([0.49,-0.22,0.055,0.45])
        bianhuan.dx = 0.1
        real_state = bianhuan.surrogate_to_real(input_surrogate)
        shishi.load()
        y1_predict = shishi.k1.predict(input_surrogate)
        y2_predict = shishi.k2.predict(input_surrogate)
        y3_predict = shishi.k3.predict(input_surrogate)
        print('MXairfoil: test the trained Surrogate', '\n omega predict = ' , y1_predict  , '\n rise predict = ' , y2_predict,  '\n turn predict = ' , y3_predict)
    elif flag == 6 :
        shishi = Surrugate(case='NACA65')
        shishi.load()
        shishi.visual_2D('chi_in','chi_out')
        # shishi.visual_2D('mxthk','umxthk')
    elif flag == 7 :
        # shishi = Surrugate(case='Rotor67',x_dim=18)
        shishi = Surrugate(case='Rotor67',x_dim=18)
        # shishi.operating_3D()
        # shishi.operating_3D2()
        # shishi.get_Pout_distribution(28000,model='dengbi')
        # shishi.get_Pout_distribution(28000,model='cifang')
        # shishi.debug_back_from_list()
        # shishi.test_repeatability_3D()
        # N_X = shishi.test_to_train()
        # N_X = shishi.train_to_test(index_begin=300,index_end=315)

        # shishi.re_calculate_geo(index=[232])
        ylist = shishi.train_model_mul_3D(300,72,model='train',pmin=[1.5,1.5,1.5,1.5,1.5,1.5,1.5],thetamax = [10,10,10,10,10,10,10])
        # ylist = shishi.train_model_mul_3D(346,72,model='train',pmin=[1.5,1.5,1.5,1.5,1.5,1.5,1.5],thetamax = [10,10,10,10,10,10,10])
        # ylist = shishi.train_model_mul_3D(625,64,model='demo180')
        shishi.test_all(15,load_model='import',X_model='X_rand',self_model='test') # import/exist, X_train/X_rand/X_merge
        shishi.save()
    elif flag == 72 :
        # this is to calculate Rotor37.
        shishi = Surrugate(case='Rotor37',x_dim=18)
        shishi.operating_3D2()
    elif flag == 8 :
        shishi = Surrugate(case='Rotor67',x_dim=18)
        # shishi.recycle_result_3D(4)
        shishi.debug_back_from_list(model= 'rescue')
    elif flag == 82:
        shishi = Surrugate(case='Rotor67',x_dim=18)
        shishi.debug_Rotor_case(model = 'CFD',location = 'E:/XXHdatas/EnglishMulu/case0CFD0')
    total_time_end = time.time()
    total_time_cost = total_time_end - total_time_start
    print('MXairfoil: total time cost ='+str(total_time_cost))
    print('MXairfoil: finish a surrogate model related process. En Taro XXH!')



