from usedImports import np
# import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
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
