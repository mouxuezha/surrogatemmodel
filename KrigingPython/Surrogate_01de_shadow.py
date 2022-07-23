from Surrogate_01de import Surrugate
from Surrogate_01de import record_progress
import time 
import numpy as np
import sys

if __name__ =='__main__':
    print('MXairfoil: test the Surrogate')
    total_time_start = time.time()
    # shishi = Surrugate(case='NACA65')
    # shishi = Surrugate(case='Rotor67',x_dim=18)
    flag = 72 # 1 for train, -1 for clear, 2 for test all. 5 for single point . 7 for 3D point, and trainning, 8 for debug.
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
        
        ylist = shishi.train_model_mul_3D(128,72,model='demo180')
        # ylist = shishi.train_model_mul_3D(625,64,model='demo180')
        shishi.test_all(50,load_model='import',X_model='X_rand',self_model='demo180') # import/exist, X_train/X_rand/X_merge
        shishi.save()
        shishi.auto_record()
    elif flag == 72 :
        # this is to calculate Rotor67 original state.
        shishi = Surrugate(case='Rotor67',x_dim=18)
        state_original_real = np.array( [0,0,0,0,0,0,0.447895,0.122988,0.064253,0.050306, -0.639794,-0.052001,-0.050454,-0.148836,0.223533,0.656313,0.965142,1.098645])
        state_0_normal = np.zeros(18)
        state_original_surrogate = shishi.bianhuan.real_to_surrogate(state_original_real)
        state_0_real = shishi.bianhuan.normal_to_real(state_0_normal)

        state_original_normal = shishi.bianhuan.real_to_normal(state_original_real)
        state_0_normal = shishi.bianhuan.real_to_normal(state_0_real)
        print('MXairfoil: En Taro XXH!')
        # shishi.operating_3D2()

    elif flag == 73 :
        shishi = Surrugate(case='Rotor67',x_dim=18)
        # ylist = shishi.train_model_mul_3D(625,64,model='demo180')
        from jiadian import jiadian
        import pickle
        
        # location_X_sample = shishi.location + '/theta10点数128/X.pkl'
        # location_Y_sample = shishi.location + '/theta10点数128/y_CFD.pkl'
        location_X_sample = shishi.location + '/theta10点数128但是sobol/X.pkl'
        location_Y_sample = shishi.location + '/theta10点数128但是sobol/y_CFD.pkl'
        location_Y_add = shishi.location + '/theta10点数64/y_CFD.pkl'
        location_X_add = shishi.location + '/theta10点数64/X.pkl'

        X_sample = pickle.load(open(location_X_sample,'rb'))
        y_sample = pickle.load(open(location_Y_sample,'rb'))
        y_add_compare = pickle.load(open(location_Y_add,'rb'))
        
        location_X = shishi.location+'/X.pkl'
        location_y = shishi.location+'/y_CFD.pkl'

        zuobi = jiadian()    
        zuobi.set_N_points(2)
        zuobi.get_sample(X_sample,y_sample)
        X_add = pickle.load(open(location_X_add,'rb'))
        X_add = zuobi.load_X_add(location_X_add)
        # X_add_real,y_add_real,index = zuobi.find_y_add_mul(X_add)
        # bijiao = y_add_real - y_add_compare[index]
        # panju = sum(sum(abs(bijiao))) / len(X_add_real)
        sys.path.append(r'C:/Users/y/Desktop/DDPGshishi/main')
        from DemoFunctions import DemoFunctions
        hanshu = DemoFunctions()

        point_max = np.ones((shishi.x_dim,)) * 0.5 
        hanshu.set_point_max(point_max)

        # X_add_real,y_add_real = zuobi.find_y_add_mul2(10)
        # X_add_real,y_add_real = zuobi.find_y_add_mul3(10)
        X_add_real = X_add
        y_add_real = y_add_compare

        # this should be zero
        # panju = zuobi.check_xyadd(X_sample,y_sample,hanshu.ToulanFunction_general)
        panju = zuobi.check_xyadd(X_add_real,y_add_real,hanshu.ToulanFunction_general)
        print('MXairfoil: panju = ' + str(panju) )

        # then merge them as X and Y and save.
        X_all = np.append(X_sample,X_add_real,axis=0)
        y_all = np.append(y_sample,y_add_real,axis=0)
        pickle.dump(X_all,open(location_X,'wb'))
        pickle.dump(y_all,open(location_y,'wb'))

        # then train surrogate model.
        N_all = len(X_all)
        if N_all != len(y_all):
            raise Exception('MXairfoil: dimension unmatch when zuobi, G!')
        ylist = shishi.train_model_mul_3D(N_all,64,model='demo180')
        shishi.test_all(50,load_model='import',X_model='X_rand',self_model='demo180') 

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

