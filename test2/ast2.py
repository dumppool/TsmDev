import sys
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PFNNParameter import PFNNParameter

FLAGS = tf.flags.FLAGS
N_GPU = 2
tf.flags.DEFINE_string('job_name','ps','name of the job, default ps')
tf.flags.DEFINE_integer('task_index',0,'index of the job, default 0')


    
def PFNNPredict(X_nn, keep_prob ):
    Xdim = 343
    Ydim = 311
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
    #with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
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


def main(_):
    mnist = input_data.read_data_sets('/home/armando/datasets/mnist', one_hot=True)

    ps = [
            'localhost:9001',  # /job:ps/task:0
         ]
    workers = [
            'localhost:9002',  # /job:worker/task:0
            'localhost:9003',  # /job:worker/task:1
            ]
    clusterSpec = tf.train.ClusterSpec({'ps': ps, 'worker': workers})

    config = tf.ConfigProto()
    config.allow_soft_placement = True

    #server = tf.train.Server(clusterSpec,
    #                         job_name=FLAGS.job_name,
    #                         task_index=FLAGS.task_index,
    #                         config=config
    #                         )

    if FLAGS.job_name=='ps':
        #print(config.device_count['GPU'])
        config.device_count['GPU']=0
        server = tf.train.Server(clusterSpec,
                                 job_name=FLAGS.job_name,
                                 task_index=FLAGS.task_index,
                                 config=config
                                 )
        server.join()
        sys.exit('0')
    elif FLAGS.job_name=='worker':
        #config.gpu_options.per_process_gpu_memory_fraction = 0.2
        X = np.load('XXX.npy')
        Y = np.load('YYY.npy')
        print(X.shape)
        print(Y.shape)
        Xdim    = X.shape[1]
        Ydim    = Y.shape[1]
        samples = X.shape[0]
        rng = np.random.RandomState(23456)
        weightDecay        = 0.0025

        batch_size         = 32
        training_epochs    = 100
        N_GPU              = 2
        total_batch        = int(samples / batch_size)
        n_epochs_print     = 2
        count_test     = 0
        num_testBatch  = np.int(total_batch * count_test)
        num_trainBatch = total_batch - num_testBatch

        I = np.arange(samples)
        rng.shuffle(I)
        print("training_batch:", num_trainBatch)
        print("test_batch:", num_testBatch)
        #====================================================================
        config.device_count['GPU']=FLAGS.task_index
        server = tf.train.Server(clusterSpec,
                                 job_name=FLAGS.job_name,
                                 task_index=FLAGS.task_index,
                                 config=config
                                 )
        is_chief = (FLAGS.task_index==0)

        worker_device='/job:worker/task:{}/gpu:{}'.format(FLAGS.task_index, FLAGS.task_index)
        device_func = tf.train.replica_device_setter(worker_device=worker_device,
                                                     cluster=clusterSpec
                                                     )
    # the default values are: ps_device='/job:ps',worker_device='/job:worker'
        with tf.device(device_func):
            X_nn = tf.placeholder(tf.float32, [None, Xdim], name='x-input')
            Y_nn = tf.placeholder(tf.float32, [None, Ydim], name='y-input')
            keep_prob = tf.placeholder(tf.float32) 
            #-------------------------------------------------------            
            global_step = tf.train.get_or_create_global_step()
            #tf.Variable(0,name='global_step',trainable=False)
            learning_rate = 0.0001
            batch_idx = FLAGS.task_index
            _x = X_nn[batch_idx * batch_size: (batch_idx+1) * batch_size]
            _y = Y_nn[batch_idx * batch_size: (batch_idx+1) * batch_size]
            reuse_variables = True
            loss_op, P0, P1, P2 = get_loss( _x, _y, keep_prob, reuse_variables)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss_op,global_step=global_step)
            accuracy_op = loss_op

        sv = tf.train.Supervisor(is_chief=is_chief,
                                 init_op = tf.global_variables_initializer(),
                                 global_step=global_step)



        with sv.prepare_or_wait_for_session(server.target) as mts:
            lstep = 0
            for epoch in range( training_epochs ):
                for step in range( int(num_trainBatch/N_GPU)):
                    index_train = I[step*batch_size*N_GPU:( step + 1) * batch_size*N_GPU]
                    batch_xs = X[index_train]
                    batch_ys = Y[index_train]
                    feed_dict = {X_nn: batch_xs, Y_nn: batch_ys, keep_prob: 0.7 }
                    _,loss,gstep = mts.run([train_op,loss_op,global_step],
                                         feed_dict=feed_dict)
                    lstep +=1
                if (epoch+1)%n_epochs_print==0:
                    print('worker={},epoch={},global_step={}, local_step={}, loss = {}'.
                          format(FLAGS.task_index,epoch,gstep,lstep,loss))
            #feed_dict={x_p:x_test,y_p:y_test}
            #accuracy = mts.run(accuracy_op, feed_dict=feed_dict)
            #print('worker={}, final accuracy = {}'.format(FLAGS.task_index,accuracy))
    sv.stop()

if __name__ == '__main__':

  tf.app.run()
