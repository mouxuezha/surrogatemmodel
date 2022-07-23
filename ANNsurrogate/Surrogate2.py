# ANN surrogate model itself.
from numpy.lib.function_base import append
import pickle

import tensorflow as tf
import numpy as np
import time
import os
import math

import matplotlib.pyplot as plt
from matplotlib import cm

class Surrugate2(object):
    def __init__(self):
        print('MXairfoil: Sorrogate model initialized')
        self.LAYER1_SIZE = 5
        self.LAYER2_SIZE = 5
        self.LEARNING_RATE = 5e-3
        self.TAU = 0.001
        self.L2 = 0.01
        self.batch_size = 64
        self.X_dim = 4
        self.y_dim = 2

        self.location = 'C:/Users/y/Desktop/ANNsurrogate'
        self.script_folder = 'C:/Users/y/Desktop/EnglishMulu/testCDA1'
        self.matlab_location = 'C:/Users/y/Desktop/MXairfoilCDA'
        self.real_obs_space_h = np.array([0.4,-0.2,0.8,8])
        self.real_obs_space_l = np.array([-0.2,-0.4,0.2,3])
        self.sess = tf.Session()
        self.time_step = 0 

        # create network
        self.x_data,self.y_target,self.loss,self.net,self.is_training,self.y_predict = self.create_ANN(self.X_dim,self.y_dim)

        self.create_training_method()

    def load_data(self):
        location = self.location

        location_X = location+'/X_save.pkl'
        self.X = pickle.load(open(location_X,'rb'))

        location_y1 = location+'/y1_save.pkl'
        y1 = pickle.load(open(location_y1,'rb'))

        location_y2 = location+'/y2_save.pkl'
        y2 = pickle.load(open(location_y2,'rb'))
        
        chicun_y = y1.shape
        y1 = y1.reshape(chicun_y[0],1)
        y2 = y2.reshape(chicun_y[0],1)
        self.Y = np.append(y1,y2,axis=1)
        print('MXairfoil: successfully loaded data')

        # # do some check.
        # # unnecessary, infact.
        # chicun_X = self.X.shape
        # if chicun_X[1] != self.X_dim:
        #     print('MXairfoil: invalid input')

    def variable(self,shape,f,mingzi):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)),name=mingzi)

    def create_ANN(self,X_dim,y_dim):
        self.X_dim = X_dim
        self.y_dim = y_dim

        x_data = tf.placeholder(shape=[None, X_dim], dtype=tf.float32,name="X_data")
        y_target = tf.placeholder(shape=[None, y_dim], dtype=tf.float32,name='y_target')

        is_training = tf.placeholder(tf.bool,name='is_trainning')

        W1 = self.variable([X_dim,self.LAYER1_SIZE],X_dim,mingzi='Weight1')
        b1 = self.variable([self.LAYER1_SIZE],X_dim,mingzi='biases1')

        # W2 = self.variable([self.LAYER1_SIZE,self.LAYER2_SIZE],self.LAYER1_SIZE+y_dim)
        # b2 = self.variable([self.LAYER2_SIZE],self.LAYER1_SIZE+y_dim)

        W2 = self.variable([self.LAYER1_SIZE,self.y_dim],self.LAYER1_SIZE+y_dim,mingzi='Weight2')
        b2 = self.variable([self.y_dim],self.LAYER1_SIZE+y_dim,mingzi='biases2')

        layer1_output = tf.nn.relu(tf.matmul(x_data,W1) + b1,name='layer1_output')
        final_output = tf.nn.relu(tf.add(tf.matmul(layer1_output, W2),b2),name='final_output')

        loss = tf.reduce_mean(tf.square(y_target - final_output),name='loss')

        return x_data,y_target,loss,[W1,b1,W2,W2,b2],is_training,final_output

    def create_training_method(self):
        # then define trainning method and other things.
        self.my_opt = tf.train.GradientDescentOptimizer(self.LEARNING_RATE)
        self.train_step = self.my_opt.minimize(self.loss)
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def train(self,rand_x,rand_y):
        #trainning one batch.

        # Now we run the training step
        self.sess.run(self.train_step, feed_dict={self.x_data: rand_x, self.y_target:rand_y})
        # We save the training loss
        temp_loss = self.sess.run(self.loss, feed_dict={self.x_data: rand_x, self.y_target: rand_y})
        # loss_vec.append(np.sqrt(temp_loss))
        return temp_loss

    def test(self,x_vals_test,y_vals_test):
        test_temp_loss = self.sess.run(self.loss, feed_dict={self.x_data: x_vals_test, self.y_target: y_vals_test})
        # test_loss.append(np.sqrt(test_temp_loss))
        return test_temp_loss

    def feed_ANN(self):
        # prepare the data set.
        
        train_indices = np.random.choice(len(self.X), round(len(self.X)*0.8), replace=False)
        test_indices = np.array(list(set(range(len(self.X))) - set(train_indices)))
        
        x_vals_train = self.X[train_indices]
        x_vals_test = self.X[test_indices]
        y_vals_train = self.Y[train_indices]
        y_vals_test = self.Y[test_indices]
        #trainning loop is here.
        loss_vec = []
        test_loss = []
        i = 0 
        test_temp_loss = 114514
        while (test_temp_loss>0.001)&(i<1000000):
            i=i+1 # for i in range(500):
            rand_index = np.random.choice(len(x_vals_train), size=self.batch_size)
            # We then select the training values
            rand_x = x_vals_train[rand_index]
            # rand_y = np.transpose([y_vals_train[rand_index]])
            rand_y = y_vals_train[rand_index]
            temp_loss = self.train(rand_x,rand_y)
            loss_vec.append(np.sqrt(temp_loss))

            #test 
            test_temp_loss =self.test(x_vals_test,y_vals_test)
            test_loss.append(np.sqrt(test_temp_loss))

            if (i+1)%50==0:
                print('MXairfoil: Generation: ' + str(i+1) + '. Loss = ' + str(temp_loss) + '. Test Loss = ' +str(test_temp_loss))

        self.time_step = i 

        plt.plot(loss_vec, 'k-', label='Train Loss')
        plt.plot(test_loss, 'r--', label='Test Loss')
        plt.title('Loss (MSE) per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        # plt.show()
        shijian = time.strftime("%Y-%m-%d", time.localtime())
        plt.savefig(self.location+'ANNtrain'+shijian+'.png',dpi=750)

    def save(self):
        self.saver=tf.train.Saver(max_to_keep=1)
        print('save surrogate-network...',self.time_step)
        location = self.location + '/saved_surrogate_networks/'
        if not(os.path.exists(location)):
            try:
                os.mkdir(location)
            except:
                print('MXairfoil: can not make dir for saveing agent. ',location) 
        name = location  + 'surrogate-network'
        try:
            self.saver.save(self.sess, name, global_step = self.time_step)
            print('successfully save the surrogate-network')
        except:
            print('fail to save the surrogate-network')

    def load(self):
        location = self.location+"/saved_surrogate_networks"
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(location)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)

    def predict(self,X):
        X = X.reshape(1,4)
        Y_jieguo = self.sess.run(self.y_predict,feed_dict={self.x_data:X,self.is_training: False})[0]
        return Y_jieguo
    
    def visual_ANN(self):
        tensorboard_location = self.location + '/tensorboard_shishi'
        writer=tf.summary.FileWriter(tensorboard_location, self.sess.graph)
        X=np.zeros((1,4))
        f=self.predict(X)
        print(f)
        writer.close()
        pass


if __name__ == '__main__':
    print('MXairfoil: testing Surrogate2, which is ANN based surrogate')
    flag = 0 
    if flag == 0 :
        shishi = Surrugate2()
        shishi.load_data()
        shishi.feed_ANN()
        shishi.save()
    elif flag == 1 :
        # load data and predict
        shishi = Surrugate2()
        shishi.load_data()
        shishi.load()
        X_input = shishi.X[0,:]
        Y_real = shishi.Y[0,:]
        Y_predict = shishi.predict(X_input)
        print('MXairfoil: use this model to do some predict. Y_real = '+str(Y_real)+'  Y_predict = ' +str(Y_predict) )
    elif flag == 2 :
        # tensorboard visualization
        shishi = Surrugate2()
        shishi.load_data()
        shishi.visual_ANN()
    else:
        print('MXairfoil: nothing happen.')

    print('MXairfoil: finish testing Surrogate2')




