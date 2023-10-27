import math, time, sys
from scipy import signal as sn


def calc_resampled_size(input_sample_rate, output_sample_rate, input_length):
    """Calculate the output size after resampling.
    :returns: integer output size, >= 1
    """
    target_size = int(math.ceil((output_sample_rate / input_sample_rate) * (input_length)))
    return max(target_size, 1)

def calculate_freq(interval):
    """ Convert interval (ms) to frequency (Hz)
    """
    freq = 1000 / interval
    if abs(freq - round(freq)) < 0.01:
        freq = round(freq)
    return freq

class Resampler:
    """ Utility class to handle resampling and logging
    """

    def __init__(self, total_samples):
        self.total_samples = total_samples
        self.ix = 0
        self.last_message = 0

    def resample(self, sample, new_length, original_length):
        # Work out the correct axis
        ds_axis = 0
        if (sample.shape[0] == 1):
            ds_axis = 1

        # Resample
        if (original_length != new_length):
            sample = sn.resample_poly(sample, new_length, original_length, axis=ds_axis)

        # Logging
        self.ix += 1
        if (int(round(time.time() * 1000)) - self.last_message >= 3000) or (self.ix == self.total_samples):
            print('[%s/%d] Resampling windows...' % (str(self.ix).rjust(len(str(self.total_samples)), ' '), self.total_samples))

            if (self.ix == self.total_samples):
                print('Resampled %d windows\n' % self.total_samples)

            sys.stdout.flush()
            self.last_message = int(round(time.time() * 1000))

        return sample