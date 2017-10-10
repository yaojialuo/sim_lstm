#Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import numpy as np
import random
from random import shuffle
import tensorflow as tf
from tensorflow.python import debug as tf_debug

# arr1=np.arange(12).reshape(2,2,3)
# print(arr1)
# t=arr1.transpose((2,1,0))
# print(t,t.shape)#3,2,2
# t2=t.transpose((1,0,2))#2,3,2
# print(t2,t2.shape)
#
# quit()
# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import rnn

NUM_EXAMPLES = 100
BIT=10

train_input = ['{0:010b}'.format(i) for i in range(2**BIT)]
#shuffle(train_input)
train_input = [map(int,i) for i in train_input]
ti  = []

for i in train_input:
    temp_list = []
    for j in i:
            temp_list.append([j])
    ti.append(np.array(temp_list))

train_input = ti

train_output = []
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    train_output.append([count/BIT])

test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]
print(train_input[1],train_output[1])
print( "test and training data loaded" )
#quit()
# shape: (1000, 10, 1)
data = tf.placeholder(tf.float32, [None, BIT,1],"tf_data") #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, 1],"tf_taget")
data_id=tf.identity(data)
target_id=tf.identity(target)
num_hidden = 12
cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)
#cell = tf.contrib.rnn.BasicLSTMCell(num_hidden,state_is_tuple=True)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)
weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.matmul(last, weight) + bias
#cross_entropy = tf.reduce_sum(tf.abs(target - tf.clip_by_value(prediction,1e-10,1.0)))
cross_entropy=tf.reduce_sum()
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
pred_rounded = tf.round( prediction*10 )/BIT
mistakes = tf.not_equal( target, pred_rounded )
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

batch_size = 10
no_of_batches = int(len(train_input) / batch_size )
epoch = 1000
debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp = train_input[ptr:ptr+batch_size]
        out = train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
        #debug_sess.run(minimize,{data: inp, target: out})

        #debug_sess.run([data_id,target_id],feed_dict={data: inp, target: out})
    print( "Epoch ",str(i) )

incorrect = sess.run(error,{data: test_input, target: test_output})
print( sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]}) )
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()