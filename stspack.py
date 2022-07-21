import pandas as pd
import numpy as np
from scipy.fftpack import rfft, rfftfreq, next_fast_len
from scipy.signal import find_peaks

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

def drop_duplicate_indices(df):
    df = df.drop_duplicates(subset='time')
    df = df.set_index('time')
    return df