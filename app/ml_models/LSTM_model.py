from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os

class uni_lstm:
    '''
    functions:
    - Train the model
    - update/create/save the model
    - predict using the most recent model
    '''
    def __init__(self, data, use_recent_model=True, **kwargs):
        '''
        :param: data list: list of univariant data to train/predict
        :param: use_recent_model bool: if True load and use the most recent model
                                        otherwise retrain and resave model
        :param:
        '''
        self.data = data
        self.TRAIN_SPLIT = kwargs['TRAIN_SPLIT'] if 'TRAIN_SPLIT' in kwargs else 200
        self.BATCH_SIZE = kwargs['BATCH_SIZE'] if 'BATCH_SIZE' in kwargs else 10
        self.BUFFER_SIZE = kwargs['BUFFER_SIZE'] if 'BUFFER_SIZE' in kwargs else 20
        self.EVALUATION_INTERVAL = kwargs['EVALUATION_INTERVAL'] if 'EVALUATION_INTERVAL' in kwargs else 5
        self.EPOCHS = kwargs['EPOCHS'] if 'EPOCHS' in kwargs else 200
        
        if 'seed' in kwargs:
            tf.random.set_seed(kwargs['seed'])

        if use_recent_model:
            filename='model.h5'
            # filename = kwargs['filename'] if 'filename' in kwargs else 'model.h5'
            self.load_model(filename)
            
        
        else:
            # Normalize the data
            self. data = self.normalize(self.data)
            
            # Create windows
            univariate_past_history = kwargs['univariate_past_history'] if 'univariate_past_history' in kwargs else 20
            univariate_future_target = kwargs['univariate_future_target'] if 'univariate_future_target' in kwargs else 0
            
            x_train, y_train = self.univariate_data(0, self.TRAIN_SPLIT, univariate_past_history, univariate_future_target) # Training
            x_test, y_test = self.univariate_data(self.TRAIN_SPLIT, None, univariate_past_history, univariate_future_target) # Testing
            
            shape = x_train.shape[-2:]
            
            # Creating training slices and shuffling them
            train_univariate = tf.data.Dataset.from_tensor_slices((x_train, y_train))
            self.train_univariate = train_univariate.cache().shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE).repeat()

            # Creating testing slices and shuffling them
            test_univariate = tf.data.Dataset.from_tensor_slices((x_test, y_test))
            self.test_univariate = test_univariate.batch(self.BATCH_SIZE).repeat()
            
            # Create the model
            self.model = self.create_model(shape)
            
            # Fit the model
            self.train_model()
            
            # Save model
            filename = 'model.h5'
            self.save_model(filename,)
            
            # Make predictions
            for x, y in self.test_univariate.take(5):
                plot = self.show_plot([x[0].numpy(), y[0].numpy(),
                            self.model.predict(x)[0]], 0, 'Simple LSTM model')
                plot.show()
            
            
            
    # Training the model
    def train_model(self):
        self.model.fit(self.train_univariate, epochs=self.EPOCHS,
                            steps_per_epoch=self.EVALUATION_INTERVAL,
                            validation_data=self.test_univariate, validation_steps=50)
    
    # Save the model to current dir
    def save_model(self, filename):
        if filename is None:
            filename = 'model.h5'
            
        self.model.save(filename)
    
    # Load the model from the current directory
    def load_model(self, filename):
        if filename is None:
            filename = 'model.h5'
        
        self.model = tf.keras.models.load_model(filename)
    
    def create_model(self, shape):
        model =  tf.keras.models.Sequential([
                    tf.keras.layers.LSTM(8, input_shape=shape),
                    tf.keras.layers.Dense(1)
                    ])
        model.compile(optimizer='adam', loss='mae')
        return model
    
    # predict the future given some data
    def predict(self, data=None):
        try:
            prediction = self.model.predict(data)
            return prediction
        except Exception as e:
            print("ERROR", e)
            
        # For testing    
        # for x, y in data.take(3):
        #     plot = self.show_plot([x[0].numpy(), y[0].numpy(),
        #                 self.model.predict(x)[0]], 0, 'Simple LSTM model')
        #     plot.show()        

    
    def univariate_data(self, start_index, end_index, history_size, target_size):
        '''
        This returns the window of time for a model to train on
        :param: start_index int : Where is inside data do you want to start creating windows
        :param: end_index int : Where do you want to end creating windows
        :param: history_size int : the size of the past window of information
        :param: target_size int: How far into the future the model needs to learn
        '''
        
        data = [] # Stores the data for the training window
        labels = [] # Stores the data for the output

        start_index = start_index + history_size
        if end_index is None:
            end_index = len(self.data) - target_size

        for i in range(start_index, end_index):
            indices = range(i-history_size, i)
            
            # Reshape data from (history_size,) to (history_size, 1)
            data.append(np.reshape(self.data[indices], (history_size, 1)))
            labels.append(self.data[i+target_size])
        return np.array(data), np.array(labels)

    def normalize(self, data):
        data_mean = data[:self.TRAIN_SPLIT].mean()
        data_std = data[:self.TRAIN_SPLIT].std()
        data = (data-data_mean)/data_std
        
        return data
        
    def unnormalize(self, data, predictions):
        '''
        the predictions must be in an array format
        :returns unnormalized predictions
        '''
        data_mean = data[:self.TRAIN_SPLIT].mean()
        data_std = data[:self.TRAIN_SPLIT].std()
        return (predictions*data_std)+data_mean
    
    # For testing the model 
    def create_time_steps(self,length):
        '''
        returns an array of numbers from -length to -1. i.e: create_timesteps(5) --> [-5, -4, -3, -2, -1]
        :param: length int: size of the timestep
        '''
        time_steps = []
        for i in range(-length, 0, 1):
            time_steps.append(i)
        return time_steps
    
    # For creating quick custom plots
    def show_plot(self,plot_data, delta, title):
        '''
        - Plots the target data (y) with the time steps (X).
        - If the size of plot_data list is bigger than 1, i.e 3, the 0th index is the history (acctual data)
            the 1st index is the True future given by univariate function
            the 2nd index is the model prediction
        :params: plot_data dataframe/series: the data to plot
        :params: delta int: how fare into the future to move
        :params: title string: the title of the plot
        :returns: A plot of plot_data
        '''
        labels = ['History', 'True Future', 'Model Prediction']
        marker = ['.-', 'rx', 'go']
        time_steps = self.create_time_steps(plot_data[0].shape[0])
        if delta:
            future = delta
        else:
            future = 0

        plt.title(title)
        for i, x in enumerate(plot_data):
            if i:
                plt.plot(future, plot_data[i], marker[i], markersize=10,
                    label=labels[i])
            else:
                plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
        plt.legend()
        plt.xlim([time_steps[0], (future+5)*2])
        plt.xlabel('Time-Step')
        return plt
    
    
if __name__ == '__main__':
    print("[TESTING]")
    import pandas_datareader.data as web
    import os
    from dotenv import load_dotenv
    
    BASEDIR = os.path.join(os.getcwd(), '../../')
    load_dotenv(os.path.join(BASEDIR, '.env'))
    
    start_date = '2017-04-04'
    end_date = '2018-04-04'
    name = 'AMZN'
    
    # Reading the data
    df = web.DataReader(name=name, data_source='quandl', start=start_date, end=end_date, access_key=os.getenv("quandle_key"))
    df['mid_data'] = (df['High']+df['Low'])/2
    
    # Creating the LSTM class
    lstm = uni_lstm(df['mid_data'].values[::-1], False,
                    TRAIN_SPLIT=210, EPOCHS=100,
                    BATCH_SIZE=10, BUFFER_SIZE=20,
                    EVALUATION_INTERVAL=5, seed=13, 
                    univariate_past_history=20,
                    univariate_future_target=0,)