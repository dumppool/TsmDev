import numpy as np
import tensorflow as tf
import PFNNParameter as PFNN
import Utils as utils
from PFNNParameter import PFNNParameter
 
import os.path

tf.set_random_seed(23456)  

utils.build_path(['data'])
utils.build_path(['training'])
utils.build_path(['training/nn'])
utils.build_path(['training/weights'])
utils.build_path(['training/model'])

#X = np.float32(np.loadtxt('./data/Input.txt'))
#Y = np.float32(np.loadtxt('./data/Output.txt'))

'''
X = np.load('XXa.npy')
Y = np.load('YYa.npy')
P = np.load('PPa.npy')
P = np.reshape(P,(X.shape[0],1))#X

X = np.concatenate((X, P), axis=1)

np.save('XXX.npy',X)
np.save('YYY.npy',Y)
'''

X = np.load('XXX.npy')
Y = np.load('YYY.npy')


print(X.shape)
print(Y.shape)


Xdim    = X.shape[1]

Ydim    = Y.shape[1]
samples = X.shape[0]

print(Xdim,Ydim)
""" Phase Function Neural Network """
"""input of nn"""



"""parameter of nn"""
rng = np.random.RandomState(23456)


##===================================================================
##===================================================================

def PFNNet(X_nn, keep_prob, reuse):
  with tf.variable_scope('PfnnDevNet', reuse = reuse):
     ####parameter of nn 
     nslices = 4                             # number of control points in phase function
     phase = X_nn[:,-1]                      #phase
     P0 = PFNNParameter((nslices, 512, Xdim-1), rng, phase, 'wb0')
     P1 = PFNNParameter((nslices, 512, 512), rng, phase, 'wb1')
     P2 = PFNNParameter((nslices, Ydim, 512), rng, phase, 'wb2')
     H0 = X_nn[:,:-1] 
     H0 = tf.expand_dims(H0, -1)       
     H0 = tf.nn.dropout(H0, keep_prob=keep_prob)

     b0 = tf.expand_dims(P0.bias, -1)      
     H1 = tf.matmul(P0.weight, H0) + b0      
     H1 = tf.nn.elu(H1)             
     H1 = tf.nn.dropout(H1, keep_prob=keep_prob) 

     b1 = tf.expand_dims(P1.bias, -1)       
     H2 = tf.matmul(P1.weight, H1) + b1       
     H2 = tf.nn.elu(H2)                
     H2 = tf.nn.dropout(H2, keep_prob=keep_prob) 

     b2 = tf.expand_dims(P2.bias, -1)       
     H3 = tf.matmul(P2.weight, H2) + b2      
     H3 = tf.squeeze(H3, -1)  
     return H3


        

def PFNNetB(X_nn, keep_prob, reuse):
  with tf.variable_scope('PfDevNet', reuse = reuse):
     ####parameter of nn 
     # b是又一个Softmax模型的参数，我们一般叫做“偏置项”（bias）。     
     Xdim = 343 
     Ydim = 311
     W0 = tf.Variable(tf.zeros([Xdim, 256]))
     b0 = tf.Variable(tf.zeros([256]))
     H0 = tf.matmul(X_nn, W0) + b0
     
     W1 = tf.Variable(tf.zeros([256, Ydim]))
     b1 = tf.Variable(tf.zeros([Ydim]))
     H1 = tf.matmul(H0, W1) + b1
     # y=softmax(Wx + b)，y表示模型的输出
     return H1


def PFNNetA(X_nn, keep_prob, reuse):
  with tf.variable_scope('PfDevNet', reuse = reuse):
     ####parameter of nn 
     # b是又一个Softmax模型的参数，我们一般叫做“偏置项”（bias）
     W0 = tf.Variable(tf.zeros([343, 256]))
     b0 = tf.Variable(tf.zeros([256]))
     H0 = tf.matmul(X_nn, W0) + b0
     
     W1 = tf.Variable(tf.zeros([256, 311]))
     b1 = tf.Variable(tf.zeros([311]))
     H1 = tf.matmul(H0, W1) + b1
     # y=softmax(Wx + b)，y表示模型的输出
     return H1


#loss = tf.reduce_mean(tf.square(Y_nn - H3))
#H3 = PFNNetA(X_nn, keep_prob, False)
#loss = tf.reduce_mean(tf.square(Y_nn - H3))

learning_rate      = 0.0001
weightDecay        = 0.0025

batch_size         = 32
training_epochs    = 150
Te                 = 10
Tmult              = 2
total_batch        = int(samples / batch_size)

 

 
#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#session
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())

#saver for saving the variables
#saver = tf.train.Saver()


#batch size and epoch
print("total_batch:", total_batch)

#randomly select training set
I = np.arange(samples)
rng.shuffle(I)


#training set and  test set
count_test     = 0
num_testBatch  = np.int(total_batch * count_test)
num_trainBatch = total_batch - num_testBatch
print("training_batch:", num_trainBatch)
print("test_batch:", num_testBatch)

   
#used for saving errorof each epoch
error_train = np.ones(training_epochs)
error_test  = np.ones(training_epochs)

def get_loss(X_nn_, Y_nn_, keep_prob, reuse_variables=None):
    # 沿用5.5节中定义的函数来计算神经网络的前向传播结果。
    with tf.variable_scope('PfDevNet', reuse = reuse_variables):
      y = PFNNetA(X_nn_, keep_prob, reuse_variables)
    loss = tf.reduce_mean(tf.square(Y_nn_ - y))
    return loss
    
# 计算每一个变量梯度的平均值。
def average_gradients(tower_grads):
    average_grads = []

    # 枚举所有的变量和变量在不同GPU上计算得出的梯度。
    for grad_and_vars in zip(*tower_grads):
        # 计算所有GPU上的梯度平均值。
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        # 将变量和它的平均梯度对应起来。
        average_grads.append(grad_and_var)
    # 返回所有变量的平均梯度，这个将被用于变量的更新。
    return average_grads

    
# 定义日志和模型输出的路径。
MODEL_SAVE_PATH = "logs_and_models/"
MODEL_NAME = "model.ckpt"

N_GPU             = 2
training_epochs   = 1
def main(argv=None): 
#start to train
   with tf.Graph().as_default(), tf.device('/cpu:0'):
          # 定义基本的训练过程
        X_nn = tf.placeholder(tf.float32, [None, Xdim], name='x-input')
        Y_nn = tf.placeholder(tf.float32, [None, Ydim], name='y-input')
        keep_prob = tf.placeholder(tf.float32)  # dropout (keep_prob) rate  0.7 on training, but should be 1 for testing

 
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        tower_grads = []
        reuse_variables = False
        # 将神经网络的优化过程跑在不同的GPU上。
        for i in range(N_GPU):
            # 将优化过程指定在一个GPU上。
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    cur_loss = get_loss(X_nn, Y_nn, keep_prob, reuse_variables)
                    # 在第一次声明变量之后，将控制变量重用的参数设置为True。这样可以
                    # 让不同的GPU更新同一组参数。
                    reuse_variables = True
                    grads = opt.compute_gradients(cur_loss)
                    print("+++++++++++++++++++++++++++++++++")
                    print(grads)
                    tower_grads.append(grads)
                    
        print("==============OKK==================")
        # 计算变量的平均梯度。
        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
            	tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)

        # 使用平均梯度更新参数。
        train_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)


        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()        
        init = tf.global_variables_initializer()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True)) as sess:
            # 初始化所有变量并启动队列。
            init.run()
            summary_writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)
            
            for epoch in range(training_epochs):             
              for i in range(num_trainBatch):
               index_train = I[i*batch_size:(i+1)*batch_size]
               batch_xs = X[index_train]
               batch_ys = Y[index_train]
               feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 0.7} 
               l, _, = sess.run([cur_loss, train_op], feed_dict=feed_dict)
               #l, _, = sess.run([cur_loss, train_op], feed_dict={X: batch_x, Y: batch_y})
               #avg_cost_train += l / num_trainBatch
               if i % 100  == 0:
                  print(i, "trainingloss:", l)
            
 

if __name__ == '__main__':
	tf.app.run()