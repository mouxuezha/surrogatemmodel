import numpy as np
from call_components import call_components
import time 
import os
import pickle

from multiprocessing import Process
import multiprocessing

class shishi_duoxiancheng:
    def __init__(self):
        print('MXairfoil: shishi_duoxiancheng model initialized')
        self.location = 'C:/Users/106/Desktop/KrigingPython/backup'
        self.script_folder = 'C:/Users/106/Desktop/EnglishMulu/testCDA1'
        self.matlab_location = 'C:/Users/106/Desktop/EnglishMulu/MXairfoilCDA'
        # self.real_obs_space_h = np.array([0.4,-0.2,0.8,8])
        # self.real_obs_space_l = np.array([-0.2,-0.4,0.2,3])
        self.real_obs_space_h = np.array([0.4,-0.22,0.55,8] )
        self.real_obs_space_l = np.array([0.3,-0.38,0.35,5])

        # self.diaoyong = call_components(self.script_folder,self.matlab_location) # no need for real calculating here, disable the jisuan.
        self.X = [] 
        self.record_mul = [[0],[0]]
    def train_test(self):
        location_X = self.location+'/X.pkl'
        self.X = pickle.load(open(location_X,'rb'))
        y = self.get_y_mul2_test(self.X,50)
        print('MXairfoil: testing multithread')
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
            time.sleep(i*1) # avoid same file name.
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
        # while 1 :
        #     print('MXairfoil: this is jisuan_mul_test, '+str(multiprocessing.current_process().pid)+'\nparents id:'+str(os.getppid()))
        #     time.sleep(7)
        X_debug = X
        X = self.norm_to_real_state(X)

        diaoyong.set_value(X[0],'chi_in')

        diaoyong.set_value(X[1],'chi_out')

        diaoyong.set_value(X[2],'mxthk')

        diaoyong.set_value(X[3],'umxthk')

        #start the calculation
        while 1 :
            print('MXairfoil: this is jisuan_mul_test, '+str(multiprocessing.current_process().pid)+'\nparents id:'+str(os.getppid()))
            # diaoyong.call_matlab()
            # diaoyong.call_IGG()
            diaoyong.call_Turbo()
        # diaoyong.call_CFView()
        try:
            omega,rise = diaoyong.get_value()
        except:
            return 0,0
        
        rizhi = 'MXairfoil: struggle to find out bug..'+'\n state is '+str(X)+'\n '+diaoyong.result_folder + '\n' + str(omega)+'   ' + str(rise)  + '\n Its X is  ' +str(X_debug)
        diaoyong.jilu(rizhi)
        return 0,0
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
    def get_dataset_for_ANN(self,N,cishu):

        X_rand = np.random.uniform(-1,1,(N,4))
        yy1 = np.zeros([N,1])
        yy2 = np.zeros([N,1])

        # y = self.get_y_mul(X_rand,30)
        # y = self.get_y_mul2(X_rand,20)
        # y = self.get_y_mul_test(X_rand,40)
        y = self.get_y_mul2_test(X_rand,40)

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

        mingling = 'taskkill /F /IM shishi_main4.exe'
        os.system(mingling) # close the error messages.      
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
    def test_call(self):
        # location = self.location
        # location_X_save = location+'/X_save.pkl'
        # X_rand = pickle.load(open(location_X_save,'rb'))

        X_rand = np.array([[-0.90760896, -0.60943294,  0.46302473, -0.06403743],
        [-0.06970121, -0.81893745,  0.60930657,  0.01500868],
        [ 0.44513524, -0.61706487,  0.39709941, -0.93798097],
        [-0.28063484,  0.62009673,  0.57534876, -0.40172574],
        [ 0.3897196 , -0.94946918,  0.79474592,  0.90242599]])


        y = self.get_y_mul(X_rand[0:5][:],2)
        y2 = self.get_y_mul(X_rand[0:5][:],2)
        y_another = self.get_y_mul(X_rand[0:5][:],3)
        y_chuanxing=np.array([[ 0.08389593,  1.04775945],[ 0.26486184,  1.03218843],[ 0.23877477,  1.03396556],[ 0.2850509 ,  1.03122774],[ 0.08417529,  1.04708148]])
        # y_chuanxing = np.zeros((5,2))
        # for i in range(5):
        #     y_chuanxing[i][0],y_chuanxing[i][1]=self.jisuan(X_rand[i][:])

        # y_chuanxing2 = np.zeros((5,2))
        # for i in range(5):
        #     y_chuanxing2[i][0],y_chuanxing2[i][1]=self.jisuan(X_rand[i][:])

        print('MXairfoil:  yy1[0]='+str(y[0])+ '  y[0]='+str(y_chuanxing[0]))
    def clear(self):
        location = self.location

        location_k1 = location+'/k1.pkl'
        location_k2 = location+'/k2.pkl'
        location_X = location+'/X.pkl'
        location_y1 = location+'/y1.pkl'
        location_y2 = location+'/y2.pkl'
        location_yCFD = location+'/y_CFD.pkl'


        try:
            os.remove(location_k1)
            os.remove(location_k2)
            os.remove(location_X)
            os.remove(location_y1)
            os.remove(location_y2)
            os.remove(location_yCFD)
        except:
            print('MXairfoil: something wrong in clearing surrogate file')

        
        try:
            self.diaoyong.clear_all(self.script_folder,self.matlab_location)
        except:
            print('MXairfoil: something wrong in clearing CFD file')

        print('MXairfoil: finish executing clear process. En Taro XXH')
    def record_child(self):
        # this is to record child process start time.
        shijian = time.strftime("%Y-%m-%d,", time.localtime()) 
        bianhao = multiprocessing.current_process().pid
        if self.record_mul[0].count(bianhao) == 0:
            self.record_mul[0].append(bianhao)


if __name__=="__main__":
    flag = 1
    shishi = shishi_duoxiancheng()
    if flag == 0:
        shishi.get_y_mul2_test(2,2)
    elif flag == 1 :
        shishi.train_test()

    print('MXairfoil:end a test process, En Taro XXH!.')


