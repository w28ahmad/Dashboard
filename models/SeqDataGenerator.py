from usedImports import np


# ! For example assume DataGeneratorSeq(train_data, 5, 5)
class DataGeneratorSeq(object):
    def __init__(self, prices, batch_size, num_unroll):
        self._prices = prices  # ? the training data
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._price_length = len(self._prices) - num_unroll  # ! 1595
        self._segments = self._price_length//batch_size  # ! 1595/5 = 319
        self._cursor = [
            offset * self._segments for offset in range(self._batch_size)]
        #! 0 319 638 ...
        print(self._cursor)

    '''
        batch_data      -- Input data x of size batch_size, equally selected from the mid_prices
        batch_labels    -- Output labels y of size batch_size, repersent the future value of the corresponding y-value
        cursor          -- treated as an iterator every time the function runs it increases all the cursor values by 1
                            Ex [0, 319, 638, 957, 1276] --> [1, 320, 639, 958, 1277]
    '''

    def next_batch(self):
        batch_data = np.zeros((self._batch_size), dtype=np.float32)
        batch_labels = np.zeros((self._batch_size), dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b]+1 >= self._price_length:
                # ? This should rarely happen but if does replace that value with a value in range (0, _price_length)
                self._cursor[b] = np.random.randint(0, (b+1)*self._segments)

            # ! train_data[0], train_data[319], train_data[638], ...
            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b] = self._prices[self._cursor[b] +
                                           np.random.randint(0, 5)]  # ! train_data[0 +2], train_data[319+5], train_data[638+1], ... not sure what this does?

            # ? Increment the self._cursor[i] values by 1, making sure that the cursor values don't exceed the price length
            self._cursor[b] = (self._cursor[b]+1) % self._price_length

        return batch_data, batch_labels

    def unroll_batches(self):
        unroll_data, unroll_labels = [], []

        for ui in range(self._num_unroll):
            price_data, label = self.next_batch()

            # ? Making a set of x, y (where x and y are arrays) pairs where x is given stock value and y is a future stock value
            unroll_data.append(price_data)
            unroll_labels.append(label)

        return unroll_data, unroll_labels

    # ? reset the cursor values to random values

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(
                0, min((b+1)*self._segments, self._price_length-1))

        print(self._cursor)
