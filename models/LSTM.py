import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from usedImports import np
from SeqDataGenerator import DataGeneratorSeq as dataGenerator
from loadData import load_data, print_data
from graphs import compare_graphs, visualize

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


def LSTM_model(D, batch_size, num_unrollings, num_nodes, n_layers, dropout):
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
                                          initializer=tf.contrib.layers.xavier_initializer()) for li in n_layers]

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
    w = tf.get_variable('w', shape=[num_nodes[-1], 1],
                        initializer=tf.contrib.layer.xavier_initializer())
    b = tf.get_variable('b', initializer=tf.random_uniform([1], -0.1, 0.1))


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
    dg = dataGenerator(train_data, 5, 5)
    u_data, u_label = dg.unroll_batches()

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
