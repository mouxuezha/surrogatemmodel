# this is to calculate result of one point, and get some tu 

from Surrogate_01de import Surrugate
from Surrogate_01de import record_progress
import time 
import numpy as np
import sys
import pickle
import shutil
from huatu import huatu
import os
from parameters import parameters_Rotor67_performance 
import threading 
from call_components import call_components
if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
    # which means in my diannao
    WEIZHI = r'C:\Users\y\Desktop\DDPGshishi\DDPG-master\DDPG-master100'
    sys.path.append(WEIZHI)
else :
    # which means in server
    WEIZHI = r'D:\XXHcode\DDPGshishi\DDPG-master\DDPG-master100'
    sys.path.append(WEIZHI)
from main_auto_post import main_auto_post


class result_process(Surrugate):
    def __init__(self, xishu_limit = 3 ,**kargs):
        super().__init__(case='Rotor67',x_dim=18,**kargs)
        print('MXairfoil: start trying to calculate and save one point')
        self.N_points = 1
        self.N_steps_per_point = 40 
        self.N_model = 7
        self.xishu_limit = xishu_limit
        
        if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
            # which means in my diannao.
            self.result_process_location = r'E:\EnglishMulu\KrigingPython\result_process'
        else:
            self.result_process_location = r'D:\XXHdatas\KrigingPython\result_process' 
        if os.path.exists(self.result_process_location):
            print('MXairfoil: result_process_location exists')
        else:
            os.mkdir(self.result_process_location)
            print('MXairfoil: result_process_location established')

    def calculate_single(self,X,N_thread=72):
        # check the X
        if len(X[0]) != self.x_dim:
            raise Exception('MXairfoil: invalid X in result_process')
        
        print('MXairfoil: X generated.')
        self.jindu = record_progress(N_points = len(X)*self.N_steps_per_point)
        # massflow_rate_lower,massflow_rate_upper,efficiency_integration,pi_integration
        
        # self.massflow_threshold = 15 # zuobi, useless
        y = self.get_y_3D_mul2(X,N_thread,model='result_process') 

        print('MXairfoil: y of selected X calculated. En Taro XXH \n'+str(y))

    def get_original_duibi(self,X_input_normal,N_thread=72):
        # x_original_surrogate = self.bianhuan.real_to_surrogate(self.state_original_real)
        # self.calculate_single(x_original_surrogate)
        geshu = len(X_input_normal)
        self.N_points=geshu
        self.calculate_single(X_input_normal,N_thread=N_thread)
        model_huatu = [] 
        index_huatu = [] 
        for i in range(geshu):
            model_huatu.append('result_process')
            index_huatu.append(i)
        model_huatu.append('train')
        index_huatu.append(0)        
        self.huatu_quxian(model=model_huatu,index=index_huatu)

    def huatu_quxian(self,model=['result_process','train'],index=[0,0]):
        y_list_list = [] 
        if len(model) != len(index):
            raise Exception('MXarifoil: invalid input in huatu_quxian, G')
        
        # then get y_list_list.
        for i in range(len(model)) : 
            if model[i] == 'train':
                location_y_list = self.location+'/y_list.pkl'
            elif model[i] == 'test':
                location_y_list = self.location+'/y_test_list.pkl'
            elif model[i] == 're_calculate':
                raise Exception('MXarifoil: invalid input in huatu_quxian, G')
            elif model[i] == 'result_process':
                location_y_list = self.location+'/y_list_result_process.pkl'
            
            y_list = pickle.load(open(location_y_list,'rb'))
            y_list_index = y_list[self.N_steps_per_point*index[i]:self.N_steps_per_point*(index[i]+1)]
            
            # tried to zuobi 
            if (i == 0) and (model[i]=='result_process'):
                self.massflow_threshold = 20 # 15.7
            else:
                self.massflow_threshold = 20 
            y_list_index = self.check_y_curve(y_list_index)
            y_list_list.append(y_list_index)
            

        # then huatu. y_list : massFlow,piStar,eta_i
        tu_eta = huatu(0)
        tu_pi =  huatu(0)
        tu_eta.set_location(self.result_process_location)
        tu_pi.set_location(self.result_process_location)
        for y_list in y_list_list:
            y_array = np.array(y_list)
            shuru_pi = np.array([y_array[:,0],y_array[:,1]])
            shuru_pi = shuru_pi.T
            shuru_pi = self.filter_y(shuru_pi)
            shuru_eta = np.array([y_array[:,0],y_array[:,2]])
            shuru_eta = shuru_eta.T
            shuru_eta = self.filter_y(shuru_eta)
            tu_pi.load_data_add(shuru_pi) 
            tu_eta.load_data_add(shuru_eta) 
        x_name = r'$\dot{m}$'
        y_name_pi = r'$\pi$'
        y_name_eta = r'$\eta$'
        tu_pi.huatu2D_mul2(x_name,y_name_pi,'flow-pressure ratio',*model,modle='all',single_ax=True,align_flag=False,x_min=25,y_min=1.5)
        tu_pi.save_all()
        tu_eta.huatu2D_mul2(x_name,y_name_eta,'flow-efficiency',*model,modle='all',single_ax=True,align_flag=False,x_min=25,y_min=0.8)
        tu_eta.save_all()
        print('MXairfoil: finished get_y_3D_mul2 for result_process')
    
    def filter_y(self,data):
        # filter points that y==0
        data_new = []
        for data_hang in data:
            if data_hang[1] == 0:
                pass
            else:
                data_new.append(data_hang)
        data_new = np.array(data_new)
        y_curve = np.flipud(data_new)
        return y_curve

    def case_X_transfer(self,location_target=None,version='V0.2'):
        if location_target == None:
            location_target = self.location
        X_old,X_add = self.X_transfer(location_target=location_target,version=version)
        geshu = len(X_add)
        kaishi = len(X_old)
        for i in range(geshu):
            self.case_transfer(model_old='result_process',model_new='train',index_geometry_old=i,index_geometry_new=kaishi+i,model_transfer='copy')

    def case_transfer(self,model_old,model_new,index_geometry_old,index_geometry_new,model_transfer='rename'):
        self.jindu = record_progress(N_points = 1*self.N_steps_per_point)
        for index_CFD in range(self.N_steps_per_point):
            flag_old ,location_old = self.jindu.check_3D_result(index_geometry = index_geometry_old,index_CFD=index_CFD,model=model_old)
            flag_new ,location_new = self.jindu.check_3D_result(index_geometry = index_geometry_new,index_CFD=index_CFD,model=model_new)
            if (flag_new == False) and (flag_old == True):
                # 
                if model_transfer == 'rename':
                    os.rename(location_old,location_new)
                elif model_transfer == 'copy':
                    shutil.copytree(location_old,location_new)
                print('MXairfoil: transfered one CFD point to ' + model_new + ' data, new geometry index=' + str(index_geometry_new))
                pass
            else:
                print('MXairfoil: Wrong when index_CFD='+str(index_CFD))            
    def X_transfer(self,location_target=None,version = None):
        jieguo  = result_jilu()
        if location_target == None:
            location_target = self.location + '/X.pkl'
        else:
            location_target = location_target + '/X.pkl'

        X_old = pickle.load(open(location_target,'rb'))    
        if version == None:   
            X_add = result_jilu().get_X_all()
        elif type(version) == str:
            X_add = result_jilu().get_X_filter(version=version)

        X_new=np.append(X_old,X_add,axis=0)
        pickle.dump(X_new,open(location_target,'wb'))
        print('MXairfoil: X_new generated')
        return X_old,X_add

    def huatu_contour_mul(self,model,index_geometry=[],index_CFD = [] ,N_thread=10,location_extra_script = None,**kargs): 
        # copy script into location, and modify it
        # print('MXairfoil: undefined yet, G')
        
        location_list_list = [] 
        self.jindu = record_progress(N_points = len(index_geometry)*len(index_CFD))
        self.location_extra_script = location_extra_script
        for i_g in index_geometry:
            for i_C in index_CFD:
                location_list = [] 
                flag ,location_stored = self.jindu.check_3D_result(index_geometry = i_g,index_CFD=i_C,model=model)
                if flag == True:
                    # which means everything is ok.
                    location_list.append(location_stored)
                    location_list.append(i_g)
                    location_list.append(i_C)
                    location_list.append(model)

        # then allocation the list.
        location_part_list = self.allocate_list(X_list=location_list,N_thread=N_thread)

        # then multi thread running.
        nloops = range(N_thread)
        threads = [] 
        for i in nloops:
            t = threading.Thread(target=self.huatu_contour_single,args=(i,location_part_list[i]))
            threads.append(t)

        #start the threads 
        for i in nloops:
            time.sleep(0.1) 
            threads[i].start()

        # waiting for the end of all threads
        for i in nloops:
            threads[i].join()
        pass
    
    def huatu_contour_single(self,N,location_part):
        diaoyong = call_components(self.script_folder,self.matlab_location,case=self.case) # index it unnecessarry.
        diaoyong.massflow_deviation_threshold = self.massflow_deviation_threshold
        diaoyong.massflow_0 = self.massflow_0
        diaoyong.efficiency_threshold=self.efficiency_threshold        
        geshu=len(location_part)
        
        for i in range(geshu):
            label = location_part[i][3]+str(location_part[i][1])+str(location_part[i][2])
            diaoyong.call_CFview_lumped(location_part[i][0],self.location_extra_script,label=label)

        print('MXairfoil:finish one thread. number =' +str(N))


    def allocate_list(self,X_list,N_thread):
        # 
        chicun = len(X_list)
        N_part = int(chicun / N_thread) # max(int(chicun / N),1 )# at least 1 

        X_part_list = [] 
        yushu = chicun - N_part*N_thread # it is 0 if perfect.
        index_begin = 0 
        index_end = 0 
        for i in range(N_thread):
            if i < yushu : 
                index_end = index_end + N_part+1
            else:
                index_end = index_end + N_part
            X_list_part = X_list[index_begin:index_end]
            index_begin = index_end + 0
        X_part_list.append(X_list_part)
        return X_part_list


class result_jilu(main_auto_post):
    def __init__(self,lujing=None) -> None:
        
        if os.environ['COMPUTERNAME'] == 'DESKTOP-GMBDOUR' :
            # which means in my diannao.
            self.result_location = r'E:\EnglishMulu\KrigingPython\result_process'
        else:
            self.result_location = r'D:\XXHdatas\KrigingPython\result_process' 
        if os.path.exists(self.result_location):
            print('MXairfoil: result_location exists')
        else:
            os.mkdir(self.result_location)
            print('MXairfoil: result_location established')
        try:
            self.result_location_result_jilu = self.result_location + '/result_jilu.pkl'
            self.data =  pickle.load(open(self.result_location_result_jilu,'rb'))
        except:
            self.data = []           
        super().__init__(weizhi = lujing)
        pass
        

    def add_result(self,X,index_agent =0, version='V0.1',model='run'):
        shijian = time.strftime("%Y-%m-%d", time.localtime())
        agent_name = 'agent0indedx' + str(index_agent)
        data_single = []

        if len(X) > 1:
            # then reshape.
            X = X.reshape(1,len(X))

        data_single.append(X)
        data_single.append(agent_name)
        data_single.append(version)
        data_single.append(shijian) 
        self.data.append(data_single)
        if model == 'debug':
            pass
        else:
            pickle.dump(self.data,open(self.result_location_result_jilu,'wb'))
    
    def get_X(self,index=0):
        print('MXairfoil: X used, agent=' + self.data[index][1]+' version=' +self.data[index][2]+'\nsaved in '+ self.data[index][3])
        return self.data[index][0]

    def get_X_all(self):
        # get all X for capability.
        geshu = len(self.data)
        X = np.zeros((geshu,18),dtype=float)
        for i in range(geshu):
            X[i] = self.data[i][0][0]
        print('MXairfoil: heres '+ str(geshu)+ ' X recorded')
        return X

    def get_X_filter(self,version='V0.1',agent='agent0'):
        geshu = len(self.data)
        X = np.zeros((geshu,18),dtype=float)  
        n=0
        for i in range(geshu):
            if (self.data[i][2].find(version) != -1) and(self.data[i][1].find(agent) != -1) :
                X[n] = self.data[i][0][0]
                n=n+1  
        print('MXairfoil: heres '+ str(n)+ ' X filtered with'+agent + ' '+version )    
        return X[0:n]

    def add_X_from_lujing(self,N_lujing = 10 ,agent_folder=None, index_agent =0, version='V0.1'):
        # just read X from lujing.
        # 0, load data.
        # model_add = 'debug'
        model_add = 'run'
        self.N_lujing = N_lujing
        agent_name = 'agent0indedx' + str(index_agent)
        location_lujing = agent_folder + '/' + version + '/' + agent_name
        lujing_list , performance_list = self.load_lujing(location =location_lujing )
        s_dim = self.real_dim
        # 1, get X_end         
        X_input_normal = self.get_state_end(folder=agent_folder + '/' + version,index=index_agent,return_model = 'normal')
        self.add_result(X_input_normal,index_agent=index_agent,version=version,model=model_add)
        # 2, get X_check
        check_index = 70 # it should be less than len(lujing_list[i])
        for i in range(self.N_lujing):
            X_check_surrogate_single  = lujing_list[i][check_index,0:s_dim]
            X_check_normal_single = self.transfer.surrogate_to_normal(X_check_surrogate_single)
            self.add_result(X_check_normal_single,index_agent=index_agent,version=version+'check'+str(check_index),model=model_add)

        check_index = 99 # it should be less than len(lujing_list[i])
        for i in range(self.N_lujing):
            X_check_surrogate_single  = lujing_list[i][check_index,0:s_dim]
            X_check_normal_single = self.transfer.surrogate_to_normal(X_check_surrogate_single)
            self.add_result(X_check_normal_single,index_agent=index_agent,version=version+'check'+str(check_index),model=model_add)

    def load_lujing(self,location=None):
        if location == None:
            location = self.agent0_location
        
        lujing = np.array([]).reshape(0,self.real_dim)
        performance = np.array([]).reshape(0,self.env.performance_dim)

        lujing_list = [] 
        performance_list = []
        
        for i in range(self.N_lujing):
            wenjianming_lujing = location +'/lujing' + str(i) + '.pkl' 
            wenjianming_performance = location + '/performance' + str(i) + '.pkl'

            lujing = pickle.load(open(wenjianming_lujing,'rb'))
            performance = pickle.load(open(wenjianming_performance,'rb'))

            lujing_list.append(lujing)
            performance_list.append(performance)
        
        return lujing_list , performance_list

    def delete_X(self,version='V0.2',save_model = False):
        n_deleted = 0 
        for index in range(len(self.data)):
            if self.data[index][2].find(version) != -1:
                self.data[index] = 'deleted'
                n_deleted = n_deleted +1
        for i in range(n_deleted):
            self.data.remove('deleted')
        print('MXairfoil: attension, '+str(n_deleted)+' of X are going to be deleted.')
        if save_model:
            pickle.dump(self.data,open(self.result_location_result_jilu,'wb'))

    def print_X(self,version='V0.2'):
        # transfer the X into real, for MATLAB huatu.
        X_normal_filtered = self.get_X_filter(version=version)
        X_end_normal = X_normal_filtered[0]
        X_end_real = self.transfer.normal_to_real(X_end_normal)
        X_end_surrogate = self.transfer.normal_to_surrogate(X_end_normal)
        print('MXairfoil: X_end of ' + version + 'printed' + '\nX_end_normal=' + str(X_end_normal) + '\nX_end_real=' + str(X_end_real) + '\nX_end_surrogate=' + str(X_end_surrogate) )
        print('===========En Taro XXH!===========')

if __name__ == '__main__':
    flag =21 # 0X for load data, 2X for duibi. 1X for transfer, 3X for transfer
    if flag == 0 :
        X_input_normal =np.array([ 0.21491229,-0.03737392,0.27456058,0.32348403,0.34222834,0.00401075
        ,0.17772984,-0.46194502,0.12570813,-0.07269927,-0.22270224,
        -0.15535595,-0.0035044,0.60471284,0.5988186,-0.30489146,-0.14960518,-0.59771533]).reshape(1,18)
        X_input_normal2 =np.array([0.0888203,0.13285155,-0.22279993,-0.28497498,0.28038848,0.14148325,0.21032072,
        0.16847188,-0.35338159,-0.23529239,-0.31991755,0.06984959,-0.43396328,-0.50457702,
        0.25619202,0.53799777,0.40465909,-0.66289491]).reshape(1,18)
        shishi = result_jilu()
        shishi.add_result(X_input_normal,index_agent=2,version='V0.1')
        shishi.add_result(X_input_normal2,index_agent=12,version='V0.1')
    if flag == 0.1:
        # load data from existing location
        agent_folder = r'E:\XXHdatas\EnglishMulu\agents'
        jieguo  = result_jilu(lujing=agent_folder)
        jieguo.delete_X(version='V0.2')
        jieguo.add_X_from_lujing(agent_folder=agent_folder,index_agent=12,version='V0.2')
        jieguo.get_X_filter(version='V0.2')
        print('MXairfoil: finish adding X from agent_folder')
    elif flag == 1 :
        shishi = result_process()
        shishi.case_transfer(model_old='undefine',model_new='result_process',index_geometry_old=0,index_geometry_new=0)
    elif flag == 11:
        shishi = result_process()
        # shishi.case_transfer(model_old='result_process',model_new='train',index_geometry_old=0,index_geometry_new=375,model_transfer='copy')
        # shishi.case_transfer(model_old='result_process',model_new='train',index_geometry_old=1,index_geometry_new=376,model_transfer='copy')
        # shishi.X_transfer()
        shishi.case_X_transfer(version='V0.2')
    elif flag ==2 :
        jieguo  = result_jilu()
        shishi = result_process()
        shishi.get_original_duibi(X_input_normal=jieguo.get_X_all(),N_thread=30)
    elif flag == 21:
        # this is big check, calculate X_ave and X_check.
        jieguo = result_jilu()
        shishi= result_process()
        shishi.get_original_duibi(X_input_normal=jieguo.get_X_filter(version='V0.2'),N_thread=75)
        
    elif flag == 3 :
        jieguo = result_jilu()
        jieguo.print_X(version='V0.2')