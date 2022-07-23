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
import time_out 
import os

import matplotlib.pyplot as plt
from matplotlib import cm

from parameters import parameters
from multiprocessing import Process
import threading 

import shutil

class Surrugate(object):
    
    def __init__(self):
        print('MXairfoil: Sorrogate model initialized')
        self.location = 'C:/Users/106/Desktop/KrigingPython'
        self.script_folder = 'C:/Users/106/Desktop/EnglishMulu/testCDA1'
        self.matlab_location = 'C:/Users/106/Desktop/EnglishMulu/MXairfoilCDA'
        self.real_obs_space_h = np.array([0.4,-0.22,0.55,8])
        self.real_obs_space_l = np.array([0.3,-0.38,0.35,5])

        self.diaoyong = call_components(self.script_folder,self.matlab_location)

        # self.sp = samplingplan(4)
        self.sp = samplingplan(2)
        print('MXairfoil: 2D debuging in init')
        self.X = [] 

    def __del__(self):
        try:
            self.diaoyong.del_gai()
            # del self.diaoyong
        except Exception as e:
            print('MXairfoil: no diaoyong object here, exit directly')
            print(e)

    def jisuan(self,X):
        X = self.norm_to_real_state(X)
        if 0 :
            omega = X[0]
            rise = X[1]
            turn = X[2]
            print('MXairfoil: attention, jisuan is debugging!')
            return omega,rise,turn

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
        omega,rise,turn = self.diaoyong.get_value_new()

        #for debug
        # omega = X[0]*X[1]
        # rise = X[2]*X[3]

        rizhi = 'MXairfoil: Surrogate model trained for one step.'+'\n state is '+str(X)
        self.diaoyong.jilu(rizhi)
        return omega,rise,turn

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
        diaoyong.call_matlab()
        diaoyong.call_IGG()
        diaoyong.call_Turbo()
        diaoyong.call_CFView()
        try:
            omega,rise,turn = diaoyong.get_value_new()
        except:
            print('XMairfoil: wrong when reaing value using get_value_new')
            os.system('pause')
            return 0,0,0
            
        

        #for debug
        # omega = X[0]*X[1]
        # rise = X[2]*X[3]
        turn = X[0]*X[1]
        

        # rizhi = 'MXairfoil: struggle to find out bug..'+'\n state is '+str(X)+'\n '+diaoyong.result_folder + '\n' + str(omega)+'   ' + str(rise)  + '\n Its X is  ' +str(X_debug)
        # diaoyong.jilu(rizhi)
        return omega,rise,turn

    def real_to_norm_state(self,state):
        # # this is change into [-1,1]
        # real_state_bili = ( self.real_obs_space_h - self.real_obs_space_l ) /2 
        # real_state_c = ( self.real_obs_space_h + self.real_obs_space_l ) /2
        # norm_state = (state - real_state_c) / real_state_bili

        # this is change into [0,1]
        real_state_bili = ( self.real_obs_space_h - self.real_obs_space_l )
        norm_state = (state - self.real_obs_space_l) / real_state_bili
        return norm_state
    
    def norm_to_real_state(self,state):
        # this is change from [-1,1] to real 
        # real_state_bili = ( self.real_obs_space_h - self.real_obs_space_l ) /2 
        # real_state_c = ( self.real_obs_space_h + self.real_obs_space_l ) /2
        # real_state = state*real_state_bili + real_state_c

        # this is change from [0,1] to real 
        real_state_bili = ( self.real_obs_space_h - self.real_obs_space_l )
        real_state = state*real_state_bili + self.real_obs_space_l
        return real_state

    def get_y(self,N,X_part):
        print('MXairfoil: this is thread '+str(N) + ', successfully started')
        chicun = X_part.shape
        if chicun[0]==0:
            #which means X is empty here.
            print('MXairfoil: empty X are fed into some threads, warning')
            return 

        zhi = np.zeros([chicun[0],3])

        # this is for calculate 
        diaoyong = call_components(self.script_folder,self.matlab_location) 
        # rizhi = '\n\n\nMXairfoil:this is thread '+str(N)+' ,begin, \n X_part is' + str(X_part) +'\n result folder:'+ diaoyong.result_folder + '\n matlab location:' + diaoyong.matlab_location
        # diaoyong.jilu(rizhi)  
        for i in range(chicun[0]):
            try:
                # zhi[i][0],zhi[i][1],zhi[i][2] = self.jisuan_mul(diaoyong,X_part[i])
                zhi[i][0],zhi[i][1],zhi[i][2] = self.jisuan_mul_2D(diaoyong,X_part[i])
                print('MXairfoil: debuging in get_y')
            except Exception as e:
                rizhi = '\n\n\nMXairfoil:this is thread '+str(N)+'\n result folder:'+ diaoyong.result_folder + '\n matlab location:' + diaoyong.matlab_location +'\n***********************\nRunning time out\n***********************\n'
                diaoyong.jilu(rizhi)  
                zhi[i][0] = 0
                zhi[i][1] = 0
                zhi[i][2] = 0
                os.system('pause')

        
        try:
            # del diaoyong
            diaoyong.del_gai()
        except Exception as e :
            print('MXairfoil: there are something wrong when del diaoyong')
            print(e)

        # # this is for debug;
        # print('MXairfoil: attention! get_y is running in debug model')
        # for i in range(chicun[0]):
        #     zhi[i][0] = X_part[i][0] 
        #     zhi[i][1] = X_part[i][1] 
        
        #then save y
        wenjianming = self.location + '/'+str(N)+'.pkl'
        
        try:
            pickle.dump(zhi,open(wenjianming,'wb'))
            print('MXarifoil: this is thread ',N,' ,finished.','results are in \n' , wenjianming,'X_part is ',X_part.shape,'\n\n\n')
        except OSError as e :
            print('MXarifoil: this is thread '+str(N)+', find an OSError here')
            print(e)
        # time.sleep(0.1)       
        self.clear_process()
    
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
    
    def get_y_mul_test(self,X,N):
        # this is to test where is the  bottleneck of multi thread 
        threads = [] 
        nloops=range(N)

        # first divide X into parts, and feed into threads.
        chicun = X.shape
        N_part = round(chicun[0] / N)

        for i in range(N-1):
            X_part = X[N_part*i:N_part*(i+1)]
            t = threading.Thread(target=self.get_y_test,args=(i,X_part))
            threads.append(t)
        #     print(i)
        # print(i)
        i = i +1 
        X_part = X[N_part*i:chicun[0]]
        t = threading.Thread(target=self.get_y_test,args=(i,X_part))
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

    def get_y_mul2_test(self,X,N):
        # it is said that multiprocessing is more powerful.
        processes = [] 
        nloops=range(N)

        # first divide X into parts, and feed into threads.
        chicun = X.shape
        N_part = round(chicun[0] / N)

        for i in range(N-1):
            X_part = X[N_part*i:N_part*(i+1)]
            t = Process(target=self.get_y_test,args=(i,X_part))
            processes.append(t)
        #     print(i)
        # print(i)
        i = i +1 
        X_part = X[N_part*i:chicun[0]]
        t = Process(target=self.get_y_test,args=(i,X_part))
        processes.append(t)

        #start the threads 
        for i in nloops:
            time.sleep(i) # avoid same file name.
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

    def get_y_test(self,N,X_part):
        chicun = X_part.shape
        if chicun[0]==0:
            #which means X is empty here.
            print('MXairfoil: empty X are fed into some threads, warning')
            return 

        zhi = np.zeros([chicun[0],2])

        # this is for calculate 
        diaoyong = call_components(self.script_folder,self.matlab_location) 
        # rizhi = '\n\n\nMXairfoil:this is thread '+str(N)+' ,begin, \n X_part is' + str(X_part) +'\n result folder:'+ diaoyong.result_folder + '\n matlab location:' + diaoyong.matlab_location
        # diaoyong.jilu(rizhi)  
        for i in range(chicun[0]):
            try:
                zhi[i][0],zhi[i][1] = self.jisuan_mul_test(diaoyong,X_part[i])
            except:
                rizhi = '\n\n\nMXairfoil:this is thread '+str(N)+'\n result folder:'+ diaoyong.result_folder + '\n matlab location:' + diaoyong.matlab_location + '\n***********************\nRunning time out\n***********************\n'
                diaoyong.jilu(rizhi)  
                zhi[i][0] = 0
                zhi[i][1] = 0

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
        except OSError:
            print('MXarifoil: this is thread '+str(N)+', find an OSError here')
        time.sleep(0.1)       

    def jisuan_mul_test(self,diaoyong,X):
        X_debug = X
        X = self.norm_to_real_state(X)

        diaoyong.set_value(X[0],'chi_in')

        diaoyong.set_value(X[1],'chi_out')

        diaoyong.set_value(X[2],'mxthk')

        diaoyong.set_value(X[3],'umxthk')

        #start the calculation
        while 1 :
            print('MXairfoil: trying to find where the bottleneck is')
            diaoyong.call_matlab()
            # diaoyong.call_IGG()
            # diaoyong.call_Turbo()
            # diaoyong.call_CFView()
        try:
            omega,rise = diaoyong.get_value()
        except:
            return 0,0
        
        # rizhi = 'MXairfoil: struggle to find out bug..'+'\n state is '+str(X)+'\n '+diaoyong.result_folder + '\n' + str(omega)+'   ' + str(rise)  + '\n Its X is  ' +str(X_debug)
        # diaoyong.jilu(rizhi)
        return 0,0
 
    def save(self):
        location = self.location

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

        print('MXairfoil: successfully saved surrogate model')

    def clear(self):
        location = self.location

        location_k1 = location+'/k1.pkl'
        location_k2 = location+'/k2.pkl'
        location_k3 = location+'/k2.pkl'
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
        
        self.diaoyong.del_gai()

        print('MXairfoil: finish executing clear process. En Taro XXH')

    def get_dataset_for_ANN(self,N,cishu):

        X_rand = np.random.uniform(0,1,(N,4))
        yy1 = np.zeros([N,1])
        yy2 = np.zeros([N,1])

        # y = self.get_y_mul(X_rand,30)
        y = self.get_y_mul2(X_rand,20)
        # y = self.get_y_mul_test(X_rand,40)
        # y = self.get_y_mul2_test(X_rand,40)

        for i in range(N):
            yy1[i][0]=y[i,0]
            yy2[i][0]=y[i,1]

        location = self.location
        location_X_save = location+'/X_save'+str(cishu)+'.pkl'
        location_y1_save = location+'/y1_save'+str(cishu)+'.pkl'
        location_y2_save = location+'/y2_save'+str(cishu)+'.pkl'

        pickle.dump(X_rand,open(location_X_save,'wb'))
        pickle.dump(yy1,open(location_y1_save,'wb'))
        pickle.dump(yy2,open(location_y2_save,'wb'))

        self.clear_process()
      
    def check_dataset_for_ANN(self,cishu):
        location = self.location
        for i in range(cishu):

            location_X_save = location+'/X_save'+str(i)+'.pkl'
            location_y1_save = location+'/y1_save'+str(i)+'.pkl'
            location_y2_save = location+'/y2_save'+str(i)+'.pkl'

            
            try:
                X_rand_cishu = pickle.load(open(location_X_save,'rb'))
                yy1_cishu = pickle.load(open(location_y1_save,'rb'))
                yy2_cishu = pickle.load(open(location_y2_save,'rb'))
            except:
                print('MXairfoil: no more parts, exit the reading loop')
                break

            if i == 0:
                X_rand = X_rand_cishu
                yy1 = yy1_cishu
                yy2 = yy2_cishu
            else:
                X_rand = np.append(X_rand,X_rand_cishu,axis=0)
                yy1 = np.append(yy1,yy1_cishu,axis=0)
                yy2 = np.append(yy2,yy2_cishu,axis=0)

        N_test = 5 
        # y = self.get_y_mul(X_rand[0:N_test][:],2)
        # y_another = self.get_y_mul(X_rand[0:N_test][:],3)
        select_index = np.random.choice(len(X_rand), N_test, replace=False)
        # X_select = X_rand[select_index]
        # yy1_select = yy1[select_index]
        # yy2_select = yy2[select_index]

        X_select = X_rand[0:5]
        yy1_select = yy1[0:5]
        yy2_select = yy2[0:5]

        yy_select = np.append(yy1_select,yy2_select,axis=1)
        y_bingxing = self.get_y_mul(X_select,2)
        # y_chuanxing = np.zeros((N_test,2))
        # for i in range(N_test):
        #     y_chuanxing[i][0],y_chuanxing[i][1]=self.jisuan(X_select[i][:])
        panju = np.sum((yy_select-y_bingxing)**2)

        print('MXairfoil: \n    yy_select[0]='+str(yy_select[0]) +'\n    y[0]='+str(y_bingxing[0]),'\n    panju = '+str(panju) + ' (it should be 0.0)')

        if panju>0.005:
            print('MXairfoil: test failed, there must be something wrong.')
        else:
            print('MXairfoil: test done. It seems right, En Taro XXH!')

    def visual_2D(self,x_name,y_name):
        
        shijian = time.strftime("%Y-%m-%d", time.localtime())
        print('MXairfoil: test visual_2D for surrogate model\n'+shijian)
        X=np.zeros((parameters.get_number(),))
        for canshu in parameters:
            value_location = self.diaoyong.matlab_location+'/input/CDA1/'+canshu.name+'.txt'
            X[canshu.value] = self.diaoyong.get_value2(value_location)
        X = self.real_to_norm_state(X)

        x1 = np.arange(0,1.01,0.01)
        x2 = np.arange(0,1.01,0.01)
        X1,X2 = np.meshgrid(x1,x2)
        Y1 = np.zeros(X1.shape)
        Y2 = np.zeros(X2.shape)

        # get some data for trainning ANN.
        # X_save = np.zeros([X1.shape[1]*X2.shape[0] , 4]) 
        X_save = np.array([])
        y1_save = np.array([])
        y2_save = np.array([])

        # get some data for trainning ANN .
        for i in range(X1.shape[1]):
            for j in range(X2.shape[0]):
                X[parameters[x_name].value] = X1[i][i]
                X[parameters[y_name].value] = X2[j][j]
                Y1[i][j] = self.k1.predict(X)
                Y2[i][j] = self.k2.predict(X)
                # Y1[i][j],Y2[i][j] = self.jisuan(X)

                # X_save.append(X, axis=0)
                if (i+j)==0:
                    X_save = X.reshape((1,parameters.get_number()))
                else:
                    X_save = np.append(X_save,X.reshape((1,parameters.get_number())), axis=0)
                y1_save=np.append(y1_save,Y1[i][j])
                y2_save=np.append(y2_save,Y2[i][j])
        
        location = self.location
        location_X_save = location+'/X_save.pkl'
        pickle.dump(X_save,open(location_X_save,'wb'))
        location_y1_save = location+'/y1_save.pkl'
        pickle.dump(y1_save,open(location_y1_save,'wb'))
        location_y2_save = location+'/y2_save.pkl'
        pickle.dump(y2_save,open(location_y2_save,'wb'))


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
        ax.set_xlabel(r'$\chi_{in} $')
        ax.set_ylabel(r'$\chi_{out} $')
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
        ax.set_xlabel(r'$\chi_{in}$')
        ax.set_ylabel(r'$\chi_{out}$')
        biaoti_rise = r'$Rise$'
        ax.set_title(biaoti_rise)
        plt.colorbar(cset1)
        plt.savefig(self.location+'/visual2Drise'+shijian+'.png',dpi=300)
        # plt.show()
    
    def clear_process(self):
        mingling1 = 'taskkill /F /IM rm.exe'
        print(mingling1)
        mingling2 = 'taskkill /F /IM cmd.exe'
        print(mingling2)
        try:
            os.system(mingling1)
            # os.system(mingling2)
        except:
            print('MXairfoil: nothing to clear.')

    def jisuan_mul_2D(self,diaoyong,X):
        # this is for 2D debug
        X = np.array([X[0],X[1],0,0])
        X = self.norm_to_real_state(X)
        print('MXairfoil: attention, jisuan_mul is debugging!')

        diaoyong.set_value(X[0],'chi_in')

        diaoyong.set_value(X[1],'chi_out')

        # diaoyong.set_value(X[2],'mxthk')

        # diaoyong.set_value(X[3],'umxthk')

        #start the calculation
        # print('MXairfoil: debugging. Do not actually call CFD components')
        diaoyong.call_matlab()
        diaoyong.call_IGG()
        diaoyong.call_Turbo()
        diaoyong.call_CFView()
        try:
            omega,rise,turn = diaoyong.get_value_new()
        except:
            print('XMairfoil: wrong when reaing value using get_value_new')
            os.system('pause')
            return 0,0,0
            
        

        #for debug
        # omega = X[0]*X[1]
        # rise = X[2]*X[3]
        turn = X[0]*X[1]
        

        # rizhi = 'MXairfoil: struggle to find out bug..'+'\n state is '+str(X)+'\n '+diaoyong.result_folder + '\n' + str(omega)+'   ' + str(rise)  + '\n Its X is  ' +str(X_debug)
        # diaoyong.jilu(rizhi)
        return omega,rise,turn

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



if __name__ =='__main__':
    print('MXairfoil: test the Surrogate')
    total_time_start = time.time()
    shishi = Surrugate()
    # shishi.clear()
    flag = 5
    # 1 for training, 2 for test, -1 for clear, 7 for visual 2d (not recommend)
    if flag ==0 :
        shishi.clear()
    elif flag == 3 :
        cishu = 200
        error_time = 0 
        for i in range(cishu):
            shishi.clear()
            try:
                shishi.get_dataset_for_ANN(400,i)
            except:
                error_time = 0 
                print('MXairfoil:something wrong when get_dataset_for_ANN, \nwrong time:'+str(error_time)+'  wrong rate: '+str(error_time/i))
        # shishi.get_dataset_for_ANN(10000)
        # get a dataset for ANN.
    elif flag == 4 :
        shishi.check_dataset_for_ANN(2)
    elif flag == 5 :
        X_gg = np.array([[ 0.14648438,  0.65429688,  0.61914062,  0.29101562],
       [ 0.42773438,  0.21289062,  0.60742188,  0.90429688],
       [ 0.09960938,  0.36132812,  0.75585938,  0.73632812]])
        X_real = shishi.norm_to_real_state(X_gg[0])
        print('MXairfoil: this is the bug input going to debug: '+str(X_real))
    elif flag == 7 :
        shishi.load()
        shishi.visual_2D('chi_in','chi_out')

    total_time_end = time.time()
    total_time_cost = total_time_end - total_time_start
    print('MXairfoil: total time cost ='+str(total_time_cost))
    print('MXairfoil: finish a surrogate model related process. En Taro XXH!')



