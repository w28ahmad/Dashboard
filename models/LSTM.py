
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from usedImports import np, plt, os, sys
from SeqDataGenerator import DataGeneratorSeq as dataGenerator
from loadData import load_data, print_data
from graphs import compare_graphs, visualize

sys.path.append(os.path.abspath(os.path.join('..', 'webapp')))
from utils.pickleUtils import write_pickle_file, read_pickle_file
# ? Scale the data between 0 and 1


def normalizer(train_data, test_data):
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1, 1)
    test_data = test_data.reshape(-1, 1)

    # Train the scaler with the training data
    smoothing_window_size = 800
    for di in range(0, 3200, smoothing_window_size):
        # print(train_data[di:di+smoothing_window_size, :])
        if (len(train_data[di:di+smoothing_window_size, :]) > 0):
            scaler.fit(train_data[di:di+smoothing_window_size, :])
            train_data[di:di+smoothing_window_size,
                       :] = scaler.transform(train_data[di:di+smoothing_window_size, :])

    # ? You normalize the last bit of remaining data, never seems to happen
    # scaler.fit(train_data[di+smoothing_window_size:,:])
    # train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

    train_data = train_data.reshape(-1)
    test_data = scaler.transform(test_data).reshape(-1)
    return (train_data, test_data)


def exponential_moving_average(train_data):
    EMA = 0.0
    gamma = 0.1
    for ti in range(0, 1600):
        EMA = gamma*train_data[ti] + (1-gamma)*EMA
        train_data[ti] = EMA

    return train_data


def standard_average_prediction(df, train_data):
    window_size = 10
    N = train_data.size
    std_average_predictions = []
    mse_error = []
    std_ave_x = []

    for pred_idx in range(window_size, N):
        date = df.loc[pred_idx, 'Date']

        std_average_predictions.append(
            np.mean(train_data[pred_idx-window_size:pred_idx]))
        mse_error.append(
            (std_average_predictions[-1] - train_data[pred_idx])**2)
        std_ave_x.append(date)

    print("MSE error for standard averageing: %.5f" % (0.5*np.mean(mse_error)))
    return(std_ave_x, std_average_predictions, range(window_size, N))

# ? Exponential average prediction


def exponential_prediction(train_data):
    # window_size = 100
    N = train_data.size

    avg_predicted_value = []
    ave_x = []

    mse_error = []

    running_mean = 0
    avg_predicted_value.append(running_mean)

    decay = 0.5

    for pred_idx in range(1, N):
        running_mean = decay*running_mean+(1-decay)*(train_data[pred_idx-1])
        avg_predicted_value.append(running_mean)
        mse_error.append((running_mean-train_data[pred_idx])**2)

    print('MSE error for EMA averaging: %.5f' % (0.5*np.mean(mse_error)))

    return (avg_predicted_value, range(0, N))

# ? Creating the LSTM model


def LSTM_model(D, batch_size, num_unrollings, num_nodes, n_layers, dropout, all_mid_data):
    # ? Defining Inputs and outputs
    # Input Data
    train_inputs, train_outputs = [], []

    # Unroll the input over the time defining placeholders for each timestep
    '''
    tf.placeholder      -- a variable that will be assigned at a later date, this allows tf to create computation graphs 
                            the need for data
                        -- shape defines a tensor shape=[3, 4] defines a matrix with 3 rows and 4 columns (so in both cases
                            essentially hava a vector because col=1)
                        -- name is just the name of the variable, used to organize operation on the tensorboard
    '''
    for ui in range(num_unrollings):
        train_inputs.append(tf.placeholder(tf.float32, shape=[
                            batch_size, D], name="train_inputs_%d" % ui))
        train_outputs.append(tf.placeholder(tf.float32, shape=[
                             batch_size, 1], name="train_output_%d" % ui))

    # ? Defining Parameters of the LSTM and Regression layer
    '''
    tf.contrib.rnn.LSTMCell -- Short long term memory cell, for more info https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43905.pdf
                            -- num_units are the number of units inside an LSTM cell that perform the same set of function
                                you can think of num_nodes as how many times the LSTM operations are going to loop
                            -- state_is_tuple If True, accepted and returned states are 2-tuples of the cell_state and memory state_state
                            -- xavier_initializer(), when assigning the weight values for each node at the start we initialize such that 
                               xavier_initializer makes sure that the variance between the x and y remains the same, we want this because
                               This helps us keep the signal from exploding to a high value or vanishing to zero.
                               https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/
                            A good image that shows the whole picture
                            https://www.oreilly.com/library/view/neural-networks-and/9781492037354/assets/mlst_1412.png
    '''
    lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=num_nodes[li], state_is_tuple=True,
                                          initializer=tf.contrib.layers.xavier_initializer()) for li in range(n_layers)]

    '''
                            -- A dropout on the input means that for a given probability, the data on the input connection to each
                               LSTM block will be excluded from node activation and weight updates. This will ultimate help to reduce
                               overfitting as the will not be extremely large, which often results in overfitting.
                            -- Dropout simulates a sparse activation from a given layer, which interestingly, in turn, encourages the
                               network to actually learn a sparse representation as a side-effect.
    '''
    drop_lstm_cells = [tf.contrib.rnn.DropoutWrapper(
        lstm, input_keep_prob=1 - dropout) for lstm in lstm_cells]

    '''
    tf.contrib.rnn.MultiRNNCell --  Stacks all the cells in the array, connection the current cells input to the next cells output
                                    This has an effect of increasing depth. This increased depth result in additional hidden layers.
                                    These additional hiddel layers are understood to recombine the learned representation from prior
                                    layers and create new representations at high levels of abstraction. For example, from lines to
                                    shapes to objects. So generalizing the results to more abstract cases.
    '''
    drop_multi_cell = tf.contrib.rnn.MultiRNNCell(drop_lstm_cells)
    multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)

    '''
    tf.get_variable             --  Creates a new variable with a given shape and initializer 
    tf.random_uniform           --  Generates a tensor with a given size and range with a uniform distribution
    '''
    # Weights and biases
    w = tf.get_variable('w', shape=[num_nodes[-1], 1],
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b', initializer=tf.random_uniform([1], -0.1, 0.1))

    # ? Calculating LSTM output and Feeding it to the regression layer to get final prediction
    # Create cell state and hidden state variables
    c, h = [], []
    initial_state = []

    '''
    give each layer a cell state and hidden state of batch_size x num_node tensor full of zeros
    trainable refers to if the variables are mutable, for example weights and baises must be trainable
    '''
    for li in range(n_layers):
        c.append(tf.Variable(
            tf.zeros([batch_size, num_nodes[li]]), trainable=False))
        h.append(tf.Variable(
            tf.zeros([batch_size, num_nodes[li]]), trainable=False))
        initial_state.append(tf.contrib.rnn.LSTMStateTuple(c[li], h[li]))

    '''
    Concatinate all the train_inputs as rows into a large matrix
    '''
    # Do several tensor transofmations, because the function dynamic_rnn requires the output to be of
    # a specific format. Read more at: https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
    all_inputs = tf.concat([tf.expand_dims(t, 0)
                            for t in train_inputs], axis=0)

    # all_outputs is [seq_length, batch_size, num_nodes
    all_lstm_output, state = tf.nn.dynamic_rnn(
        drop_multi_cell, all_inputs, initial_state=tuple(initial_state), time_major=True, dtype=tf.float32)

    # reshape all_lstm_outputs to (batch_size*num_unrollings x 150)
    all_lstm_output = tf.reshape(
        all_lstm_output, [batch_size*num_unrollings, num_nodes[-1]])

    all_outputs = tf.nn.xw_plus_b(all_lstm_output, w, b)
    split_outputs = tf.split(all_outputs, num_unrollings, axis=0)

    # ? Loss Calculation and Optimizer
    # When calculating the loss you need to be careful about the exact form, because you calculate
    # loss of all the unrolled steps at the same time
    # Therefore, take the mean error or each batch and get the sum of that over all the unrolled steps

    print("Defining Loss")
    loss = 0.0

    with tf.control_dependencies([tf.assign(c[li], state[li][0]) for li in range(n_layers)]
                                 + [tf.assign(h[li], state[li][0]) for li in range(n_layers)]):

        for ui in range(num_unrollings):
            loss += tf.reduce_mean(0.5 *
                                   (train_inputs[ui] - split_outputs[li]) ** 2)

    print("Learning rate decay operations")
    global_step = tf.Variable(0, trainable=False)
    inc_gstep = tf.assign(global_step, global_step + 1)
    tf_learning_rate = tf.placeholder(shape=None, dtype=tf.float32)
    tf_min_learning_rate = tf.placeholder(shape=None, dtype=tf.float32)

    learning_rate = tf.maximum(tf.train.exponential_decay(
        tf_learning_rate, global_step, decay_steps=1, decay_rate=0.5, staircase=True), tf_min_learning_rate)

    # Optimizer
    print('TF Optimization operations')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer = optimizer.apply_gradients(zip(gradients, v))

    print('\t All Done')

    # Predicting related calculations
    print("Defining related prediction TF functions")

    sample_inputs = tf.placeholder(tf.float32, shape=[1, D])

    # Mainting LSTM state for prediction stage
    sample_c, sample_h, initial_sample_state = [], [], []
    for li in range(n_layers):
        sample_c.append(tf.Variable(
            tf.zeros([1, num_nodes[li]]), trainable=False))
        sample_h.append(tf.Variable(
            tf.zeros([1, num_nodes[li]]), trainable=False))
        initial_sample_state.append(
            tf.contrib.rnn.LSTMStateTuple(sample_c[li], sample_h[li]))

    reset_sample_states = tf.group(*[tf.assign(sample_c[li], tf.zeros([1, num_nodes[li]])) for li in range(n_layers)],
                                   *[tf.assign(sample_h[li], tf.zeros([1, num_nodes[li]])) for li in range(n_layers)])

    sample_outputs, sample_state = tf.nn.dynamic_rnn(multi_cell, tf.expand_dims(sample_inputs, 0),
                                                     initial_state=tuple(
        initial_sample_state),
        time_major=True,
        dtype=tf.float32)

    with tf.control_dependencies([tf.assign(sample_c[li], sample_state[li][0]) for li in range(n_layers)] +
                                 [tf.assign(sample_h[li], sample_state[li][1]) for li in range(n_layers)]):
        sample_prediction = tf.nn.xw_plus_b(
            tf.reshape(sample_outputs, [1, -1]), w, b)
    print("\t All done")

    # Running the LSTM model
    epochs = 30
    valid_summary = 1  # Interval you make the test predictions

    n_predict_once = 5  # Number of steps you continously want to predict for
    train_seq_length = train_data.size  # the full size of training the data

    train_mse_ot = []  # Accumulate train losses
    test_mse_ot = []  # Accumulate the test losses
    predictions_over_time = []  # Accumulate predictions

    mse_history = []

    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Used for decaying the learning the learning rate
    loss_nondecrease_count = 0
    # If the test error has not decreased in this many steps, decrease the learning rate
    loss_nondecrease_threshold = 2

    print("Initialized")
    average_loss = 0

    # Define the data generator
    data_gen = dataGenerator(train_data, batch_size, num_unrollings)
    x_axis_seq = []

    # Points you start your test predictions
    test_points_seq = np.arange(3000, 3200, 5).tolist()

    for ep in range(epochs):

        # ========================= Training =====================================
        for step in range(train_seq_length//batch_size):

            u_data, u_labels = data_gen.unroll_batches()

            feed_dict = {}
            for ui, (dat, lbl) in enumerate(zip(u_data, u_labels)):
                feed_dict[train_inputs[ui]] = dat.reshape(-1, 1)
                feed_dict[train_outputs[ui]] = lbl.reshape(-1, 1)

            feed_dict.update(
                {tf_learning_rate: 0.0001, tf_min_learning_rate: 0.000001})

            _, l = session.run([optimizer, loss], feed_dict=feed_dict)

            average_loss += l

        # ============================ Validation ==============================
        if (ep+1) % valid_summary == 0:

            average_loss = average_loss / \
                (valid_summary*(train_seq_length//batch_size))

            # The average loss
            if (ep+1) % valid_summary == 0:
                print('Average loss at step %d: %f' % (ep+1, average_loss))

            train_mse_ot.append(average_loss)

            average_loss = 0  # reset loss

            predictions_seq = []

            mse_test_loss_seq = []

            # ===================== Updating State and Making Predicitons ========================
            for w_i in test_points_seq:
                mse_test_loss = 0.0
                our_predictions = []

                if (ep+1)-valid_summary == 0:
                    # Only calculate x_axis values in the first validation epoch
                    x_axis = []

                # Feed in the recent past behavior of stock prices
                # to make predictions from that point onwards
                for tr_i in range(w_i-num_unrollings+1, w_i-1):
                    current_price = all_mid_data[tr_i]
                    feed_dict[sample_inputs] = np.array(
                        current_price).reshape(1, 1)
                    _ = session.run(sample_prediction, feed_dict=feed_dict)

                feed_dict = {}

                current_price = all_mid_data[w_i-1]

                feed_dict[sample_inputs] = np.array(
                    current_price).reshape(1, 1)

                # Make predictions for this many steps
                # Each prediction uses previous prediciton as it's current input
                for pred_i in range(n_predict_once):

                    pred = session.run(sample_prediction, feed_dict=feed_dict)

                    our_predictions.append(np.asscalar(pred))

                    feed_dict[sample_inputs] = np.asarray(pred).reshape(-1, 1)

                    if (ep+1)-valid_summary == 0:
                        # Only calculate x_axis values in the first validation epoch
                        x_axis.append(w_i+pred_i)

                    mse_test_loss += 0.5*(pred-all_mid_data[w_i+pred_i])**2

                session.run(reset_sample_states)

                predictions_seq.append(np.array(our_predictions))

                mse_test_loss /= n_predict_once
                mse_test_loss_seq.append(mse_test_loss)

                if (ep+1)-valid_summary == 0:
                    x_axis_seq.append(x_axis)

            current_test_mse = np.mean(mse_test_loss_seq)

            # Learning rate decay logic
            if len(test_mse_ot) > 0 and current_test_mse > min(test_mse_ot):
                loss_nondecrease_count += 1
            else:
                loss_nondecrease_count = 0

            if loss_nondecrease_count > loss_nondecrease_threshold:
                session.run(inc_gstep)
                loss_nondecrease_count = 0
                print('\tDecreasing learning rate by 0.5')

            test_mse_ot.append(current_test_mse)
            print('\tTest MSE: %.5f' %
                  np.mean(mse_test_loss_seq))
            mse_history.append(mse_test_loss_seq)
            predictions_over_time.append(predictions_seq)
            print('\tFinished Predictions')

    # replace this with the epoch that you got the best results when running the plotting code
    best_prediction_epoch = mse_history.index(min(mse_history))+1
    print("Best Epoch: ", best_prediction_epoch)

    write_pickle_file("lstm.pkl", [predictions_over_time, x_axis_seq, best_prediction_epoch], "picklefiles")

def visualize_predictions(df, all_mid_data, predictions_over_time, x_axis_seq, best_prediction_epoch):
    plt.figure(figsize = (18,18))
    # plt.subplot(2,1,1)
    # plt.plot(range(df.shape[0]),all_mid_data,color='b')

    # # Plotting how the predictions change over time
    # # Plot older predictions with low alpha and newer predictions with high alpha
    # start_alpha = 0.25
    # alpha  = np.arange(start_alpha,1.1,(1.0-start_alpha)/len(predictions_over_time[::3]))
    # for p_i,p in enumerate(predictions_over_time[::3]):
    #     for xval,yval in zip(x_axis_seq,p):
    #         plt.plot(xval,yval,color='r',alpha=alpha[p_i])

    # plt.title('Evolution of Test Predictions Over Time',fontsize=18)
    # plt.xlabel('Date',fontsize=18)
    # plt.ylabel('Mid Price',fontsize=18)
    # plt.xlim(11000,12500)

    # plt.subplot(2,1,2)

    # Predicting the best test prediction you got
    plt.plot(range(df.shape[0]),all_mid_data,color='b')
    for xval,yval in zip(x_axis_seq,predictions_over_time[best_prediction_epoch]):
        plt.plot(xval,yval,color='r')

    plt.title('Best Test Predictions Over Time',fontsize=18)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.xlim(3000,3250)
    plt.show()


if __name__ == "__main__":
    data_source = "kaggle"  # kaggle or alphavantage
    df = load_data(data_source)
    # print_data(df)
    # visualize(df)

    # ? Train Test Split
    high_prices = df.loc[:, "High"].values
    low_prices = df.loc[:, "Low"].values
    mid_prices = (high_prices+low_prices)/2

    # ? Split the train test split with a 50:50 ratio
    train_data = mid_prices[:1600]
    test_data = mid_prices[1600:]

    # ? Normalize and smooth the data
    train_data, test_data = normalizer(train_data, test_data)
    train_data = exponential_moving_average(train_data)

    # ? For visualization
    all_mid_data = np.concatenate([train_data, test_data], axis=0)

    # ? Standard_average_predictions
    # dates, predictions, predict_range = standard_average_prediction(df, train_data)
    # compare_graphs(all_mid_data, range(df.shape[0]), "True", predictions, predict_range, "Prediction", "Date", "Mid Price")

    # ? Exponential_average_predictions
    # predictions, predict_range = exponential_prediction(all_mid_data)
    # print(len(predictions),len(predict_range))
    # compare_graphs(all_mid_data, range(df.shape[0]), "True", predictions, predict_range, "Prediction", "Date", "Mid Price")

    # ? Data Generator
    # dg = dataGenerator(train_data, 5, 5)
    # u_data, u_label = dg.unroll_batches()

    '''
    LSTM MODEL
    '''
    # ? Defining Hyperparameters
    # The dimention of the data, since we are looking at 1D data the dimention is 1D
    D = 1
    # The number of times you look into the future, generally the higher the better (the number of batches in the input)
    num_unrollings = 50
    # The number of samples in a batch
    batch_size = 500
    # The number of hidden nodes in each layer of the LSTM stack we are using
    num_nodes = [200, 200, 150]
    # number of layers
    n_layers = len(num_nodes)
    # the drop out rate
    dropout = 0.2
    # This is important in case you run this multiple times
    tf.reset_default_graph()

    # ? Run the LSTM model
    # LSTM_model(D, batch_size, num_unrollings,
    #            num_nodes, n_layers, dropout, all_mid_data)

    # ? Graphing the predictions
    predictions_over_time, x_axis_seq, best_prediction_epoch = read_pickle_file("lstm.pkl", "picklefiles")
    visualize_predictions(df, all_mid_data,predictions_over_time, x_axis_seq, best_prediction_epoch)