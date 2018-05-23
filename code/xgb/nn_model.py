"""
@Authors Leo.cui
7/5/2018
RF-NN Model

"""

from __future__ import print_function
import tensorflow as tf
import numpy as np

# Parameters
learning_rate = 1e-4
num_steps = 1000
display_step = 100
epochs = 10
train_proportion = 0.8 # n% data feed into model (1-n)% remain as test
npy_files = 10
model_path = "/home/leo/ant_leo/model/save_restore/ant_model"

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 128 # 2nd layer number of neurons
n_hidden_3 = 128 # 2nd layer number of neurons
num_input =  #  data input features * No.trees
num_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
rf_X = tf.placeholder("float", [None, num_input])
rf_Y = tf.placeholder("float", [None,num_classes])

#new initialize
weights = {
    'h1': tf.Variable(tf.truncated_normal([num_input, n_hidden_1], stddev = 0.1, dtype = tf.float32) , name = 'h1'),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev = 0.1, dtype = tf.float32) , name = 'h2'),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev = 0.1, dtype = tf.float32) , name = 'h3'),
    'out': tf.Variable(tf.truncated_normal([n_hidden_3, num_classes], stddev = 0.1, dtype = tf.float32) , name = 'h_out')
}
biases = {
    'b1': tf.Variable(tf.random_uniform([n_hidden_1], 0, 0.01, dtype = tf.float32) , name = 'b1'),
    'b2': tf.Variable(tf.random_uniform([n_hidden_2], 0, 0.01, dtype = tf.float32) ,name = 'b2'),
    'b3': tf.Variable(tf.random_uniform([n_hidden_3], 0, 0.01, dtype = tf.float32) , name ='b3'),
    'out': tf.Variable(tf.random_uniform([num_classes], 0, 0.01, dtype = tf.float32) , name = 'b_out')
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 128 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Hidden fully connected layer with 128 neurons
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['out']), biases['out']))
    return out_layer

# Construct model
logits = neural_net(rf_X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=rf_Y))
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)  #AdamOptimizer

train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
pred_probas = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(rf_Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

#split each npy file to train and test
def format_data(train_RF_feature, train_proportion):
    #Get the feature from the RF and feed to the NN
    batch_x = np.array([i[0] for i in train_RF_feature])
    #print(batch_x.shape)
    batch_y = np.array([i[1] for i in train_RF_feature])

    train_x = batch_x[ : int(batch_x.shape[0]*train_proportion)]
    #print(train_x.shape)
    train_y = batch_y[ : int(batch_x.shape[0]*train_proportion)]

    test_x = batch_x[int(batch_x.shape[0]*train_proportion)+1 : ]
    test_y = batch_y[int(batch_x.shape[0]*train_proportion)+1 : ]
    #print(test_x.shape)
    return train_x, train_y, test_x, test_y

#create saver instance
saver = tf.train.Saver()
count = 0

# Start training
with tf.Session() as sess:

    sess.run(init)

    for epoch in range(epochs):

        for f_num in range(0,npy_files):

            npy_path = "/home/leo/ant_leo/data/xgb_nn_{}.npy".format(f_num)
            print("Load -------------------- : xgb_nn_{}.npy".format(f_num))
            train_RF_feature = np.load(npy_path)
            train_x, train_y, test_x, test_y = format_data(train_RF_feature, train_proportion)

            for step in range(1, num_steps+1):

                #Define Total steps
                total_step = step + (f_num-1)*1000 + count
                # Run optimization op (backprop)
                sess.run(train_op, feed_dict={rf_X: train_x, rf_Y: train_y})
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc, pred = sess.run([loss_op, accuracy, pred_probas], feed_dict={rf_X: train_x,rf_Y: train_y})

                    print("Step " + str(total_step) + ", Loss= " + \
                          "{:.4f}".format(loss) + ", Training Accuracy= " + \
                          "{:.3f}".format(acc) + ", epoch {}".format(epoch))

            saver.save(sess, model_path, global_step = total_step)
            print("******** Model saved in step : %i ********\n" %(step+(f_num-1)*1000))

            print("Optimization Finished!")

            print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={rf_X: test_x,
                                              rf_Y: test_y}))
        count = total_step
        #save model in every n files
        if epoch % 1 == 0:
            #save model
            save_path = saver.save(sess, model_path)
            print("Model saved in file: %s" % save_path)
