import xgboost as xgb
from ant_function import *
import tensorflow as tf
import os
import timeit
import pandas as pd


class Xgb_Model(object):
    """docstring for ."""
    def __init__(self,
                 max_depth,
                 num_trees,
                 etc,
                 subsample,
                 colsample_bytree,
                 min_child_weight,
                 gamma,
                 model_path,
                 model_name,
                 data_path,
                 data_name,
                 **kwargs):

        #super(, self).__init__()
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.etc = etc
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.model_path = model_path
        self.model_name = model_name
        self.data_path = data_path
        self.data_name = data_name
        self.kwargs = kwargs

        for name, value in kwargs.items():
            setattr(self, name, value)

    #make self to dict
    def values(self):
        return self.__dict__

    def set_xgb_mode(self, mode):

        mode_list = ["train", "test"]

        if mode == "train":
            self.mode = mode
            return self.mode

        elif mode == "test":
            self.mode = mode
            return self.mode
        else:
            raise ValueError("****** Only accept train or test mode ******")


    def _feature_extraction(self, bst, ddata, label):

        _, _total_leaf_index = bst.get_score(importance_type='index')

        _leaf = bst.predict(ddata, pred_leaf = True)

        xgb_features = [] # data = train + label

        conncatenate_list = []

        for row in range(len(_leaf)):

            for i, x in enumerate(_total_leaf_index):

                leaf_index = _leaf[row][i] #leaf_node_index_in_each_tree

                list = [1 if n == leaf_index else 0 for n in x]

                conncatenate_list = conncatenate_list + list

            if self.mode == "train":

                xgb_features.append([int(label[row]), conncatenate_list])

                conncatenate_list = [] #flush conncatenate_list

            elif self.mode == "test" :

                xgb_features.append(conncatenate_list)

                conncatenate_list = []

        if self.mode == "train" :

            return xgb_features, len(xgb_features[0][1])

        else:

            return xgb_features


    def _save_feature_extraction(self, xgb_features, data_path, data_name):

        print("write file start ")

        if self.mode == "train":
            # TODO:  change csv
            with open(data_path + data_name + ".csv", 'a') as f:

                for row in xgb_features:

                    f.write(str(row[0])+",")

                    for i in range(len(row[1])):

                        f.write(str(row[1][i])+",")

                    f.write("\n")


        elif self.mode == "test":

            with open(data_path + data_name + "_" + self.mode + ".csv", 'a') as f:

                for row in xgb_features:

                    for i in range(len(row)):

                        f.write(str(row[i])+",")

                    f.write("\n")

        print("write file done ")

    #This method aimed to feed the train data to the xgb model to get the features.csv
    #save input to nn_input.txt file
    def batch_reader(self, batch_path, function):

        obj_dir = os.listdir(batch_path)
        for file in obj_dir:
            if file.endswith(".npy") :

                print("\nloading file : %s" %(file))
                data = np.load(batch_path + file)

                if self.mode == "train":

                    train_batch = data[:,1:]
                    label = data[:,0]
                    nn_input = function(train_batch, label)
                    #ant_xgb.load_mode(train_batch, label, mod)

                elif self.mode == "test" :

                    function(data)
                    #ant_xgb.load_mode(data, mod)

        if self.mode == "train":
            #save features_numbers to txt
            with open(self.data_path + "nn_input.txt", 'w') as f:
                f.write(str(nn_input))


    def file_check(self):

        obj_dir_model = os.listdir(self.model_path)

        if self.model_name + ".model" in obj_dir_model:

            print("\n******We found file : <{}> in path , does it need to retain ?******\n".format(self.model_name + ".model"))
            # TODO: cmd check
            check = input("Y/N ? :").lower()

            if check in ["y", "yes"] :

                check = input("Are you sure ? Y/N:").lower()

                if check in ["y", "yes"] :
                    return True

                elif check in ("n", "no", "false"):

                    return False

            elif check in ("n", "no"):

                return False

            else:

                print("\nWrong Input Please Check")
                return self.file_check()

        elif self.model_name + ".model" not in obj_dir_model:
            #train_xgb
            return True

        
    #use to train xgboost and save model
    def train_xgb(self, train_data, label):
        #initialize params
        params = self.values()

        #Load train data
        dtrain = xgb.DMatrix(train_data, label) #missing data
        bst = xgb.train(params, dtrain, self.num_trees)
        if self.model_name.endswith(".model"):
            raise ValueError("Model name can not with suffix")
        else:
            bst.save_model(self.model_path + self.model_name + ".model")

    #train_batch to reduce the computational pressure for the computer
    #use to load xgb_model and feed train data into it (feature_extraction)
    def load_mode(self, batch_path, label=None):

        ddata = xgb.DMatrix(batch_path)

        # model checking
        if self.model_name.endswith(".model"):

            bst = xgb.Booster(model_file = self.model_path + self.model_name)

        else:
            bst = xgb.Booster(model_file = self.model_path + self.model_name + ".model")

        print("loaded xgb model")

        if self.mode == "train" :

            xgb_features, nn_input = self._feature_extraction(bst, ddata, label)

            self._save_feature_extraction(xgb_features, self.data_path, self.data_name)

            print("nn_num_input ,....." , nn_input)

            return nn_input

        else:

            xgb_features = self._feature_extraction(bst, ddata, label)

            self._save_feature_extraction(xgb_features, self.data_path, self.data_name)

            return


class Neural_Network(object):

    def __init__(self,
                 lr,
                 num_classes,
                 epochs,
                 validation,
                 num_steps,
                 display_step,
                 batch_size,
                 num_threads,
                 min_after_dequeue,
                 capacity,
                 model_path,
                 model_name,
                 data_path,
                 data_name,
                 #tfrecords_filename,
                 *args,
                 **kwargs):

        #super(, self).__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.epochs = epochs
        self.validation = validation
        self.num_steps = num_steps
        self.display_step = display_step
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.min_after_dequeue = min_after_dequeue
        self.capacity = capacity
        self.model_path = model_path
        self.model_name = model_name
        self.data_path = data_path
        self.data_name = data_name
        #self.tfrecords_filename = tfrecords_filename
        self.args = args
        self.kwargs = kwargs

    @staticmethod
    def get_nn_input(data_path):

        with open(data_path + "nn_input.txt", 'r') as f:

            nn_input = f.read()

            if len(nn_input) >= 6:
                raise OverflowError(
                'Input dimension error could not feed into NN, please check the NN input'
                )

            return int(nn_input)

    def generate_tfrecords(self):

        input_filename = self.data_path + self.data_name + ".csv"
        #print("input_filename", input_filename)
        output_filename = self.data_path + self.data_name + ".tfrecords"

        for filename in os.listdir(self.data_path):

            if filename == self.data_name + ".csv":

                print("\nStart to convert {} to {}\n".format(input_filename, output_filename))
                start_time = timeit.default_timer()
                writer = tf.python_io.TFRecordWriter(output_filename)

                for line in open(input_filename, "r"):
                    data = line.split(",")

                    label = int(data[0])

                    features = [int(i) for i in data[1:-1]] #the last digit from the csv file is "/n" so [1:-1] get rid of it

                    #convert data to bytes

                    example = tf.train.Example(features = tf.train.Features(feature= {
                            ## TODO:  #int64_list, bytes_list
                            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),

                            "features": tf.train.Feature(int64_list=tf.train.Int64List(value = features))

                             }))

                    writer.write(example.SerializeToString()) #SerializeToString

                writer.close()

                print("Successfully convert {} to {}".format(input_filename,
                                                             output_filename))
                end_time = timeit.default_timer()

                print("\nThe pretraining process ran for {0} minutes\n".format(round(((end_time - start_time) / 60 ),2)))

    def w_and_b(self, num_input):

        # TODO:  change here
        # Network Parameters
        n_hidden_1 = 256 # 1st layer number of neurons
        n_hidden_2 = 128 # 2nd layer number of neurons
        n_hidden_3 = 128 # 2nd layer number of neurons

        weights = {
            'h1': tf.Variable(tf.truncated_normal([num_input, n_hidden_1], stddev = 0.1, dtype = tf.float32) , name = 'h1'),
            'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev = 0.1, dtype = tf.float32) , name = 'h2'),
            'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev = 0.1, dtype = tf.float32) , name = 'h3'),
            'out': tf.Variable(tf.truncated_normal([n_hidden_3, self.num_classes], stddev = 0.1, dtype = tf.float32) , name = 'h_out')
        }
        biases = {
            'b1': tf.Variable(tf.random_uniform([n_hidden_1], 0, 0.01, dtype = tf.float32) , name = 'b1'),
            'b2': tf.Variable(tf.random_uniform([n_hidden_2], 0, 0.01, dtype = tf.float32) ,name = 'b2'),
            'b3': tf.Variable(tf.random_uniform([n_hidden_3], 0, 0.01, dtype = tf.float32) , name ='b3'),
            'out': tf.Variable(tf.random_uniform([self.num_classes], 0, 0.01, dtype = tf.float32) , name = 'b_out')
        }

        return weights, biases

    def neural_net(self, x, weights, biases):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
        # Hidden fully connected layer with 128 neurons
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
        # Hidden fully connected layer with 128 neurons
        layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
        # Output fully connected layer with a neuron for each class
        out_layer = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['out']), biases['out']))

        return out_layer

    #pares tfrecord file
    def _pares_tf(self, serialized_data, num_input):
    #pares serialized_example
        features = {
                    'label' : tf.FixedLenFeature([], tf.int64),
                    'features' : tf.FixedLenFeature([num_input], tf.int64)  #the shape of the data
                    }

        parsed_data = tf.parse_single_example(serialized_data ,features)

        return parsed_data['features'], parsed_data['label']

    #read and rdecode tfrecord file
    def _read_and_decode(self, tfrecords_filename, num_input, shuffle_batch = True):

        #pip the data in the queue shuffle or not
        filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs= self.epochs, shuffle = False)

        reader = tf.TFRecordReader()

        _, serialized_data = reader.read(filename_queue)

        _features , _label =  self._pares_tf(serialized_data, num_input)

        #one-hot label
        _label = tf.cast(_label, tf.int32)
        #one-hot label
        _label = tf.one_hot(_label, self.num_classes, 1, 0)

        if shuffle_batch :

            #Training data in a shuffled way
            _features , _label = tf.train.shuffle_batch([_features, _label], batch_size = self.batch_size,
                                        capacity = self.capacity, num_threads= self.num_threads,  min_after_dequeue = self.min_after_dequeue)

        elif not shuffle_batch:
            _features , _label = tf.train.batch([_features, _label], batch_size = self.batch_size,
                                            capacity = self.capacity ,num_threads= self.num_threads) #min_after_dequeue = min_after_dequeue

        return _features, _label


    def run_model(self, num_input, shuffle_batch = False):

        tfrecords_filename = self.data_path + self.data_name + ".tfrecords"

        _features, _label = self._read_and_decode(tfrecords_filename, num_input, shuffle_batch = shuffle_batch)

        #split tensor to train and test set
        _train_x, _test_x = tf.split(_features, [self.batch_size - self.validation, self.validation], 0)

        _train_y, _test_y = tf.split(_label, [self.batch_size - self.validation, self.validation], 0)


        #new initialize
        global_steps = tf.Variable(0, trainable = False)
        rf_X = tf.placeholder("float", [None, num_input])
        rf_Y = tf.placeholder("float", [None, self.num_classes])
        weights, biases = self.w_and_b(num_input)

        # Construct model
        logits = self.neural_net(rf_X, weights, biases)
        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=rf_Y))
        optimizer = tf.train.AdagradOptimizer(learning_rate = self.lr)  #AdamOptimizer

        train_op = optimizer.minimize(loss_op, global_step = global_steps)

        # Evaluate model (with test logits, for dropout to be disabled)
        pred_probas = tf.nn.softmax(logits)
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(rf_Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        #create saver instance
        saver = tf.train.Saver()

        # Initialize the variables (i.e. assign their default value) in case the epochs doesn't work
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # Start training
        with tf.Session() as sess:

            #start a Session
            sess.run(init_op)
            #initialize Coordinator
            coord = tf.train.Coordinator()

            #start queue
            threads = tf.train.start_queue_runners(coord=coord)

            max_acc = 0

            f = open(self.model_path + "accuracy_log.txt", "w")

            try:

                obj_dir = os.listdir(self.model_path)

                if "checkpoint" in obj_dir :

                    print("\n******We found file : <{}> in path , keep retaining ?******\n".format("checkpoint"))

                    check = input("Y/N ? :").lower()

                    if check in ("y", "yes") :

                        check = input("Are you sure ? Y/N:").lower()

                        if check in ["y", "yes"] :

                            print("<{}> existing, strat retraining......".format(self.model_name))

                        elif check in ("n", "no"):

                            return

                    elif check in ("n", "no"):
                        print ("Processing next")
                        return

                    else:

                        print("Wrong Input")

                    saver.restore(sess, tf.train.latest_checkpoint(self.model_path))

                else :

                    print("model does not exist......")

                    sess.run(init_op)

                while not coord.should_stop():

                    train_x, train_y, test_x, test_y = sess.run([_train_x, _train_y, _test_x, _test_y])

                    # Run optimization op (backprop)
                    sess.run(train_op, feed_dict={rf_X: train_x, rf_Y: train_y})

                    if sess.run(global_steps) % self.display_step == 0 or sess.run(global_steps) == 1:

                        # Calculate batch loss and accuracy
                        loss, acc, pred = sess.run([loss_op, accuracy, pred_probas], feed_dict={rf_X: train_x,rf_Y: train_y})

                        print("Step " + str(sess.run(global_steps)) + ", Loss= " + \
                              "{:.4f}".format(loss) + ", Training Accuracy= " + \
                              "{:.3f}".format(acc))

                        print("Testing Accuracy:", \
                		sess.run(accuracy, feed_dict={rf_X: test_x, rf_Y: test_y}))

                        f.write(str(sess.run(global_steps)) + ', val_acc: ' + str(acc) + '\n')
                        #save max steps
                        if acc > max_acc:

                            max_acc = acc

                            saver.save(sess, self.model_path + self.model_name, global_step = sess.run(global_steps))

                            print("******** Model saved in step : %i ********\n" %(sess.run(global_steps)))

                print("Optimization Finished!")


            except tf.errors.OutOfRangeError:

                print("training done\n")

            finally:

                #save_path = saver.save(sess, self.model_path + self.model_name)

                #print("Model saved in file: %s" % save_path)
                f.close()

                coord.request_stop()

                coord.join(threads)

    def evl_mode(self, num_input):

        #clear the graph super important !!!!
        tf.reset_default_graph()

        test_file = self.data_path + self.data_name + "_test.csv"

        n_hidden_1 = 256 # 1st layer number of neurons
        n_hidden_2 = 128 # 2nd layer number of neurons
        n_hidden_3 = 128 # 2nd layer number of neurons
        #num_input = self.num_input  #  data input features * No.leaf node
        #num_classes = 2 # MNIST total classes (0-9 digits)its) # TODO: chane here
        preds = []
        # tf Graph input
        X = tf.placeholder("float", [None, num_input])

        weights, biases = self.w_and_b(num_input)

        #Initialize the saver : import_meta_graph
        #saver = tf.train.import_meta_graph(model_path + 'ant_model.meta')
        saver = tf.train.Saver() # Must initialize after the weight, bias ans input

        filename_queue = tf.train.string_input_producer([test_file], num_epochs= 1)
        #read data in queue
        reader = tf.TextLineReader()

        _, value = reader.read(filename_queue)

        record_defaults = [[1] for col in range(num_input+1)] #features input + 1 ("/n" from the csv file)

        cols = tf.decode_csv(value, record_defaults=record_defaults)

        features = tf.stack([cols[:-1]]) # maybe need to get rid of -1 , stack all the column together

        # Construct model
        #logits = neural_net(X)
        logits = self.neural_net(X, weights, biases)

        pred_probas = tf.nn.softmax(logits)

        init_op = tf.group(tf.local_variables_initializer())

        # Start training
        with tf.Session() as sess:
            sess.run(init_op)
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            #Restore model use latest_checkpoint
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))

            try:
                while not coord.should_stop():
                #for i in range(2):
                    test_x = sess.run(features)
                    #Instance probability
                    probs = sess.run(logits, feed_dict={X: test_x})
                    preds.append(probs[0][1])

                    if len(preds) % 5000 == 0:

                        print("{} have been done......".format(len(preds)))

            except tf.errors.OutOfRangeError:

                print("prediction done :>")

            finally:

                coord.request_stop()
                coord.join(threads)
                return preds

    @staticmethod
    def save_score(score_path, as_path, preds):

        #score_path = "/home/leo/ant_project/score/"
        #as_path = "/home/leo/ant_project/score/answer_sheet.csv"
        ## TODO:  change in the feature
        answer_sheet = pd.read_csv(as_path)
        #Dataframe data
        answer_sheet = pd.DataFrame(answer_sheet)
        #Feed result in score column
        answer = answer_sheet.assign(score = preds)
        #Save to .csv
        answer.to_csv(score_path + "score.csv" , index = None, float_format = "%.9f")

        print("Score saved to %s \n" % (score_path+ "score_xgb_{}.csv".format("after_xgb")))


    """
class Model_Evl(Neural_Network):

    def __init__(self,
                 num_classes,
                 num_input,
                 model_path,
                 data_path,
                 data_name,
                 *args):
        #super().__init__()
        self.num_classes = num_classes
        self.num_input = num_input
        self.model_path = model_path
        self.data_path = data_path
        self.data_name = data_name
        self.args = args


    def w_and_b(self, num_input):

        return super().w_and_b(num_input)

    def neural_net(self, X, weights, biases):

        return super().neural_net(X, weights, biases)


    ## TODO:  change to function

    def model_online(self):

        #clear the graph super important !!!!
        tf.reset_default_graph()

        test_file = self.data_path + self.data_name + "_test.csv"

        n_hidden_1 = 256 # 1st layer number of neurons
        n_hidden_2 = 128 # 2nd layer number of neurons
        n_hidden_3 = 128 # 2nd layer number of neurons
        num_input = self.num_input  #  data input features * No.leaf node
        #num_classes = 2 # MNIST total classes (0-9 digits)its) # TODO: chane here
        preds = []
        # tf Graph input
        X = tf.placeholder("float", [None, num_input])

        weights, biases = self.w_and_b(self.num_input)

        #Initialize the saver : import_meta_graph
        #saver = tf.train.import_meta_graph(model_path + 'ant_model.meta')
        saver = tf.train.Saver() # Must initialize after the weight, bias ans input



        filename_queue = tf.train.string_input_producer([test_file], num_epochs= 1)
        #read data in queue
        reader = tf.TextLineReader()

        _, value = reader.read(filename_queue)

        record_defaults = [[1] for col in range(num_input+1)] #features input + 1 ("/n" from the csv file)

        cols = tf.decode_csv(value, record_defaults=record_defaults)

        features = tf.stack([cols[:-1]]) # maybe need to get rid of -1 , stack all the column together

        # Construct model
        #logits = neural_net(X)
        logits = self.neural_net(X, weights, biases)

        pred_probas = tf.nn.softmax(logits)

        init_op = tf.group(tf.local_variables_initializer())


        # Start training
        with tf.Session() as sess:
            sess.run(init_op)
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            #Restore model use latest_checkpoint
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))

            try:
                while not coord.should_stop():
                #for i in range(2):
                    test_x = sess.run(features)
                    #Instance probability
                    probs = sess.run(logits, feed_dict={X: test_x})
                    preds.append(probs[0][1])

                    if len(preds) % 5000 == 0:
                        print("{} have been done......".format(len(preds)))

            except tf.errors.OutOfRangeError:

                print("prediction done :>")

            finally:


                score_path = "/home/leo/ant_project/score/"
                as_path = "/home/leo/ant_project/score/answer_sheet.csv"
                ## TODO:  change in the feature
                answer_sheet = pd.read_csv(as_path)
                #Dataframe data
                answer_sheet = pd.DataFrame(answer_sheet)
                #Feed result in score column
                answer = answer_sheet.assign(score = preds)
                #Save to .csv
                answer.to_csv(score_path + "score_xgb_{}.csv".format("after_xgb"), index = None, float_format = "%.9f")

                print("Score saved to %s \n" % (score_path+ "score_xgb_{}.csv".format("after_xgb")))

                coord.request_stop()
                coord.join(threads)
                return preds

    @staticmethod
    def save_score(score_path, as_path, preds):

        #score_path = "/home/leo/ant_project/score/"
        #as_path = "/home/leo/ant_project/score/answer_sheet.csv"
        ## TODO:  change in the feature
        answer_sheet = pd.read_csv(as_path)
        #Dataframe data
        answer_sheet = pd.DataFrame(answer_sheet)
        #Feed result in score column
        answer = answer_sheet.assign(score = preds)
        #Save to .csv
        answer.to_csv(score_path + "score.csv" , index = None, float_format = "%.9f")

        print("Score saved to %s \n" % (score_path+ "score_xgb_{}.csv".format("after_xgb")))
    """
