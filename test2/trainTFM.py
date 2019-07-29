import numpy as np
import tensorflow as tf
import PFNNParameter as PFNN
import Utils as utils
from PFNNParameter import PFNNParameter
import time
import os.path

tf.set_random_seed(23456)  

utils.build_path(['data'])
utils.build_path(['training'])
utils.build_path(['training/nn'])
utils.build_path(['training/weights'])
utils.build_path(['training/model'])

##==============================================================================
##==============================================================================


    
def PFNNPredict(X_nn, keep_prob ):
    nslices = 4                  # number of control points in phase function
    phase = X_nn[:,-1]           #phase
    rng = np.random.RandomState(1234)
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
    return H3, P0, P1, P2
    

# 定义损失函数。对于给定的训练数据、正则化损失计算规则和命名空间，计算在这个命名空间
# 下的总损失。之所以需要给定命名空间是因为不同的GPU上计算得出的正则化损失都会加入名为
# loss的集合，如果不通过命名空间就会将不同GPU上的正则化损失都加进来。
def get_loss(X_nn, Y_nn, keep_prob, reuse_variables=None):
    # 沿用5.5节中定义的函数来计算神经网络的前向传播结果。
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        Y_pred, P0, P1, P2 = PFNNPredict(X_nn, keep_prob)
    # 计算交叉熵损失。
    loss   = tf.reduce_mean(tf.square(Y_nn - Y_pred))
    return loss, P0, P1, P2 

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

X = np.load('XXX.npy')
Y = np.load('YYY.npy')

print(X.shape)
print(Y.shape)

Xdim    = X.shape[1]
Ydim    = Y.shape[1]
samples = X.shape[0]

rng = np.random.RandomState(23456)



#learning_rate      = 0.0001
weightDecay        = 0.0025

batch_size         = 32
training_epochs    = 50
N_GPU              = 2
total_batch        = int(samples / batch_size)

count_test     = 0
num_testBatch  = np.int(total_batch * count_test)
num_trainBatch = total_batch - num_testBatch

I = np.arange(samples)
rng.shuffle(I)
print("training_batch:", num_trainBatch)
print("test_batch:", num_testBatch)

# 主训练过程。
def main(argv=None): 
    # 将简单的运算放在CPU上，只有神经网络的训练过程放在GPU上。
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # 定义基本的训练过程
        #x, y_ = get_input()
        X_nn = tf.placeholder(tf.float32, [None, Xdim], name='x-input')
        Y_nn = tf.placeholder(tf.float32, [None, Ydim], name='y-input')
        keep_prob = tf.placeholder(tf.float32)  
        #regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
        
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        #learning_rate = tf.train.exponential_decay(
        #    LEARNING_RATE_BASE, global_step, 60000 / BATCH_SIZE, LEARNING_RATE_DECAY)       
        learning_rate = 0.0001
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        
        tower_grads = []
        reuse_variables = False
        # 将神经网络的优化过程跑在不同的GPU上。
        for i in range(N_GPU):
            # 将优化过程指定在一个GPU上。
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('GPU_%d' % i) as scope:
                    
                    _x = X_nn[i * batch_size: (i+1) * batch_size]
                    _y = Y_nn[i * batch_size: (i+1) * batch_size]
                    cur_loss, P0, P1, P2 = get_loss( _x, _y, keep_prob, reuse_variables)
                    # 在第一次声明变量之后，将控制变量重用的参数设置为True。这样可以
                    # 让不同的GPU更新同一组参数。
                    reuse_variables = True
                    grads = opt.compute_gradients(cur_loss)
                    tower_grads.append(grads)
        
        # 计算变量的平均梯度。
        grads = average_gradients(tower_grads)
        for grad, var in grads:
            if grad is not None:
            	tf.summary.histogram('gradients_on_average/%s' % var.op.name, grad)

        # 使用平均梯度更新参数。
        train_op = opt.apply_gradients(grads, global_step=global_step)

        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()        
        init = tf.global_variables_initializer()
        with tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True)) as sess:
            # 初始化所有变量并启动队列。
            init.run()
            summary_writer = tf.summary.FileWriter(MODEL_SAVE_PATH, sess.graph)
            
            for epoch in range(training_epochs):
              for step in range( int(num_trainBatch/N_GPU)):
                # 执行神经网络训练操作，并记录训练操作的运行时间。
                start_time = time.time()
                index_train = I[step*batch_size*N_GPU:( step + 1) * batch_size*N_GPU]
                batch_xs = X[index_train]
                batch_ys = Y[index_train]
                feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 0.7 }
                l, _, = sess.run([cur_loss, train_op], feed_dict=feed_dict)
                #==========================================================
                duration = time.time() - start_time
                
                # 每隔一段时间数据当前的训练进度，并统计训练速度。
                if step != 0 and step % 100 == 0:
                      print(i, "trainingloss:", l)
                if(step % 1000 == 0):
                    PFNN.save_network((sess.run(P0.alpha), sess.run(P1.alpha), sess.run(P2.alpha)), 
                      (sess.run(P0.beta), sess.run(P1.beta), sess.run(P2.beta)), 
                      50, 
                      'training/nn'
                      )
        
if __name__ == '__main__':
   tf.app.run()

'''
loss = get_loss(X_nn, Y_nn, keep_prob, False)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op  = optimizer.minimize(loss)
#session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#saver for saving the variables
saver = tf.train.Saver()


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

#start to train
print('Learning start..')
for epoch in range(training_epochs):
    avg_cost_train = 0
    avg_cost_test  = 0

    for i in range(num_trainBatch):
        index_train = I[i*batch_size:(i+1)*batch_size]
        batch_xs = X[index_train]
        batch_ys = Y[index_train]
        feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 0.7 }
        l, _, = sess.run([loss, train_op], feed_dict=feed_dict)
        avg_cost_train += l / num_trainBatch
        
        if i % 100 == 0:
            print(i, "trainingloss:", l)

    #print and save training test error 
    print('Epoch:', '%04d' % (epoch + 1), 'trainingloss =', '{:.9f}'.format(avg_cost_train))
    print('Epoch:', '%04d' % (epoch + 1), 'testloss =', '{:.9f}'.format(avg_cost_test))
 
      
print('Learning Finished!')
#-----------------------------above is model training----------------------------------
'''











