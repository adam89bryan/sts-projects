import pandas as pd
import numpy as np
import glob
from os import listdir
from scipy.fftpack import rfft, rfftfreq, next_fast_len
from scipy.signal import find_peaks

def drop_duplicate_indices(df):
    df = df.drop_duplicates(subset='time')
    df = df.set_index('time')
    return df

def load_modes(path, id):
    df_modes = pd.read_csv(path + '\\sites\\' + id + '\\modes\\modes.csv', index_col='Mode')
    df_modes['Upper Bound'] = df_modes['Frequency (Hz)'].rolling(2).mean().shift(-1)
    return df_modes

def load_shapes(path, id):
    modes_path = path + '\\sites\\' + id + '\\modes'
    num_shapes = len(glob.glob(modes_path + '/shape*.csv'))
    shapes = {}
    for i in range(num_shapes):
        shapes[i+1] = pd.read_csv(path + '\\sites\\' + id + '\\modes\\shapes' + str(i + 1) + '.csv')
    return shapes

def get_modes(df, df_modes):
    df['Mode'] = df_modes[df['Frequency'] < df_modes['Upper Bound']][0]
    return df

def fft_df(df, field_name, dt):

    arr = df.loc[:, [field_name]]
    arr = arr[field_name].to_numpy()

    yf = rfft(arr, next_fast_len(len(arr)))
    xf = rfftfreq(next_fast_len(len(arr)), dt)

    power = yf * np.conj(yf) / next_fast_len(len(arr))
    freq = xf

    Y = np.abs(power)
    threshold = 3*np.std(Y) + np.mean(Y)    # 3 x Standard Deviation + Arithmetic Mean
    distance = (len(freq)/max(freq))*0.25    # Number of horizontal points per 0.25 Hz

    peaks, _ = find_peaks(power, threshold = threshold, distance = distance)

    arrPower = Y[peaks]
    arrFrequency = freq[peaks]

    # arrContribution = np.empty(len(arrPower))
    # sum_power = sum(arrPower)

    # arrContribution = arrPower / sum_power

    # arrResults = np.column_stack((arrPower, arrFrequency, arrContribution))
    # print(type(arrResults))
    # print(arrResults)

    df = pd.DataFrame()
    df['Power'] = arrPower
    df['Frequency'] = arrFrequency
    df['Contribution'] = df['Power'] / df['Power'].sum()

    return df