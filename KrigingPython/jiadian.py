import numpy as np 
import pickle 


class jiadian(object):

    def __init__(self,N_points=3,X = None,y=None) -> None:
        self.N_points = N_points # find N_points points that near the X
        self.X = X 
        self.y = y 
        pass

    def set_N_points(self,N):
        # to define how many points are going to be used.
        self.N_points = N 

    def get_sample(self,X,y):
        self.X = X 
        self.y = y         
        pass

    def load_X_add(self,wenjianming='.\X.pkl'):
        # load an existing X_add
        X_add =  pickle.load(open(wenjianming,'rb'))
        return X_add

    def find_y_add(self,X_add):
        # find points that near the X
        N_samples = len(self.X)
        X_dim = len(self.X[0])
        y_dim = len(self.y[0])
        # then calculate jvli.
        jvli = np.zeros(N_samples)
        for i in range(N_samples):
            chazhi = X_add - self.X[i] 
            jvli[i] = np.linalg.norm(chazhi,ord=1)

        # then get index.
        index =[] 
        X_find = np.zeros((self.N_points,X_dim))
        y_find = np.zeros((self.N_points,y_dim))
        jvli_find = np.zeros(self.N_points)
        for i in range(self.N_points):
            weizhi = np.argmin(jvli)
            index.append(weizhi)
            
            X_find[i] = self.X[weizhi]
            y_find[i] = self.y[weizhi]
            jvli_find[i] = jvli[weizhi] 
            jvli[weizhi] = 777777777777

        # then check
        if not(self.fangxiang_check(X_add,X_find)):
            return 'G'

        # then weighted average
        jvli_find = jvli_find + np.finfo(float).eps
        jvli_fenzhiyi = 1/jvli_find 
        quanzhong = 1/jvli_find/np.sum(jvli_fenzhiyi)
        quanzhong = quanzhong.reshape((1,self.N_points))
        y_add = np.matmul(quanzhong,y_find)
        # return 0,0
        return y_add

    def find_X_add(self,X_rand):
        # find X_add. X_rand->X_find->X_add

        N_samples = len(self.X)
        X_dim = len(self.X[0])
        y_dim = len(self.y[0])
        # calculate jvli.
        jvli = np.zeros(N_samples)
        for i in range(N_samples):
            chazhi = X_rand - self.X[i] 
            jvli[i] = np.linalg.norm(chazhi,ord=1)

        # then get index.
        index =[] 
        X_find = np.zeros((self.N_points,X_dim))
        y_find = np.zeros((self.N_points,y_dim))
        jvli_find = np.zeros(self.N_points)
        for i in range(self.N_points):
            weizhi = np.argmin(jvli)
            index.append(weizhi)
            
            X_find[i] = self.X[weizhi]
            y_find[i] = self.y[weizhi]
            jvli_find[i] = jvli[weizhi] 
            jvli[weizhi] = 777777777777
        X_add = np.mean(X_find,axis=0) 
        return X_add 

    def find_y_add_mul(self,X_add):
        # for more than one X_add. En Taro XXH.
        N_X_add  = len(X_add)
        # y_add = np.zeros((N_X_add,len(self.y[0]))) 
        X_add_real = np.zeros((0,len(self.X[0])))
        y_add_real = np.zeros((0,len(self.y[0])))
        index = [] 
        for i in range(N_X_add):
            y_add_i = self.find_y_add(X_add[i])
            if y_add_i != 'G':
                y_add_real = np.append(y_add_real,y_add_i,axis=0)
                X_add_real = np.append(X_add_real,X_add[i].reshape(1,len(self.X[0])),axis=0)
                index.append(i)
        return X_add_real,y_add_real,index

    def find_y_add_mul2(self,N_excepted):
        # 1buzuo, 2buxiu. in for a penney, in for a pound 
        n=0
        x_dim = len(self.X[0])
        X_add_real = np.zeros((0,len(self.X[0])))
        y_add_real = np.zeros((0,len(self.y[0])))

        while n<N_excepted:
            X_rand = np.random.rand(1,x_dim)
            # X_rand = np.random.randn()
            y_add_i = self.find_y_add(X_rand)
            if y_add_i != 'G':
                y_add_real = np.append(y_add_real,y_add_i,axis=0)
                X_add_real = np.append(X_add_real,X_rand.reshape(1,x_dim),axis=0)
                n=n+1
                print('MXairfoil: jiadian runnning, n=' + str(n))
        
        return X_add_real,y_add_real

    def find_y_add_mul3(self,N_excepted):
        # randmly sample several points and calculate. cao.
        n=0
        x_dim = len(self.X[0])
        X_add_real = np.zeros((0,len(self.X[0])))
        y_add_real = np.zeros((0,len(self.y[0])))
        while n<N_excepted:
            X_rand = np.random.rand(1,x_dim)
            X_add_i = self.find_X_add(X_rand)

            # index_i = np.random.choice(len(self.X),size=self.N_points)
            # X_sample_rand = self.X[index_i]
            # X_add_i = np.mean(X_sample_rand,axis=0)
            y_add_i = self.find_y_add(X_add_i)
            if y_add_i != 'G':
                y_add_real = np.append(y_add_real,y_add_i,axis=0)
                X_add_real = np.append(X_add_real,X_add_i.reshape(1,x_dim),axis=0)
                n=n+1
                print('MXairfoil: jiadian runnning, n=' + str(n))
        
        return X_add_real,y_add_real        
    def check_xyadd(self,X_add,y_add,funtion):
        # from DemoFunctions import DemoFunctions
        self.diaoyong = funtion
        y_add2 = y_add * 1.0
        dianshu = len(X_add)
        for i in range(dianshu):
           y_add2[i] = funtion(X_add[i])
        # ToulanFunction_general 
        bijiao = y_add - y_add2
        panju = sum(sum(abs(bijiao))) / dianshu
        return panju

    def fangxiang_check(self,X_add,X_find):
        # check the fangxiang of X_finds, to decide if this X_add is valid.
        panju= np.zeros(self.N_points)
        v_find = X_find - X_add
        for i in range(self.N_points):
            n_one = v_find[i] / np.linalg.norm(v_find[i])
            v_find_rest = np.sum(v_find,axis=0) - v_find[i]
            n_rest = v_find_rest / np.linalg.norm(v_find_rest)
            panju[i] = np.vdot(n_one,n_rest)
        panju_zhi = np.sum(panju) / self.N_points 
        if self.N_points == 2:
            yuzhi = -0.8
        elif self.N_points == 3 :
            yuzhi = -0.9
        elif self.N_points == 4 :
            yuzhi = -0.8
        else:
            yuzhi =0 

        if panju_zhi < yuzhi:
            return True # good 
        else:
            return False # G

    def debug_jiadian(self,location='C:/Users/y/Desktop/KrigingPython'):

        location_X = location+'/X.pkl'
        self.X = pickle.load(open(location_X,'rb'))
        location_y = location+'/y_CFD.pkl'
        self.y = pickle.load(open(location_y,'rb'))

        # X_add = np.zeros(18) + 0.1 
        # X_add = self.X[0]
        # y_add = self.find_y_add(X_add)

        # X_add = self.X[0:3]
        # y_add = self.find_y_add_mul(X_add)

        X_add = self.load_X_add(r'C:\Users\y\Desktop\KrigingPython\theta10点数64/X.pkl')
        X_add_real,y_add_real,index = self.find_y_add_mul(X_add)

        print('MXairfoil: finish debugging jiadian class, En Taro XXH!')
        pass

if __name__ == '__main__':
    shishi = jiadian(N_points=4)
    shishi.debug_jiadian()


