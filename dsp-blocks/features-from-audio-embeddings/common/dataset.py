import numpy as np
import sys

class Dataset:
    '''Create an iterable dataset when x data is flattened, handling reshaping and resampling'''

    def __init__(self, X_all, metadata, axis, returns_interval=True, resample_interval=None):
        self.ix = 0
        self.returns_interval = returns_interval

        X_all_shaped = []
        y_all = []
        intervals_all = []
        current_offset = 0

        # Prepare for resampling data
        if resample_interval is not None:
            sys.path.append('/')
            from common.sampling import calc_resampled_size, calculate_freq, Resampler

            resample_utility = Resampler(len(metadata))
            target_freq = calculate_freq(resample_interval)
            intervals_all.append(resample_interval)

        # Reshape all samples
        for ix in range(len(metadata)):
            # Get x data using offset
            cur_len = metadata[ix]
            X_full = X_all[current_offset : current_offset + cur_len]
            current_offset = current_offset + cur_len

            # Split the interval, label from the features
            interval_ms = X_full[0]
            y = X_full[1]
            X = X_full[2:]

            # Reshape
            len_adjusted = cur_len - 2
            rows = int(len_adjusted / axis)
            # Data length is unexpected
            if not ((len_adjusted % axis) == 0):
                raise ValueError('Sample length is invalid, check the axis count.')

            X = np.reshape(X, (rows, axis))

            # Resample data
            if resample_interval is not None:
                # Work out the up and down factors using sample lengths
                original_length = X.shape[0]
                original_freq = calculate_freq(interval_ms)
                new_length = calc_resampled_size(original_freq, target_freq, original_length)

                # Resample
                X = resample_utility.resample(X, new_length, original_length)
            else:
                intervals_all.append(interval_ms)

            X_all_shaped.append(X)
            y_all.append(y)

        self.X_all = X_all_shaped
        self.y_all = y_all
        self.intervals = intervals_all

    def reset(self):
        self.ix = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ix >= len(self.y_all):
            self.reset()
            raise StopIteration

        X = self.X_all[self.ix]
        y = self.y_all[self.ix]
        if (len(self.intervals) == 1):
            # Resampled data has the same interval so we only store it once
            interval_ms = self.intervals[0]
        else:
            interval_ms = self.intervals[self.ix]

        self.ix += 1

        if (self.returns_interval):
            return X, y, interval_ms
        else:
            return X, y
