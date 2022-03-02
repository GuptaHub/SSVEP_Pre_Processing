import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import time


filename = 'OpenBCI-RAW-2021-11-15_Twelve.txt'
df = pd.read_csv(filename)
# df = df.loc[:, ' EXG Channel 0': ' EXG Channel 7']
df = df.loc[:, ' EXG Channel 0': ' EXG Channel 3']
print(df)


data = df.to_numpy()
data = np.transpose(data)
# print(data)

# ch_names = ['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5',
#                 'EXG Channel 6', 'EXG Channel 7']

ch_names = ['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3']

sfreq = 250
info = mne.create_info(ch_names, sfreq, ch_types='eeg')

data = data.astype(float)
# print(data)

raw = mne.io.RawArray(data[:,251:], info)
# print(raw)
# print(raw.info)

window_size = sfreq//2
temp = 250

sliding_window = []

# print("Length of Raw",len(raw[0][0][0]))
raw_len = len(raw[0][0][0]) + 250

print("Raw_len", raw_len)

### Implement Sliding Window
for sample in range(250,raw_len):
    if sample != 250 and sample % window_size == 0:
        # print("raw",raw[:])
        raw_temp = mne.io.RawArray(data[:,temp:sample],info)
        sliding_window.append(raw_temp)
        temp = sample
## Last window (less than 125 samples)
# elif raw_len == sample-1:
#     raw_temp = mne.io.RawArray(data[:,temp:sample],info)
#     sliding_window.append(raw_temp)
#     temp = sample
#     break


# raw.plot(block = True, scalings=dict(mag=1e-12, grad=4e-11, eeg=130, eog=150e-6, ecg=5e-4,
#  emg=1e2, ref_meg=1e-12, misc=1e-3, stim=1,
#  resp=1, chpi=1e-4, whitened=1e2))


# sliding_window[0].plot(block = True, scalings=dict(mag=1e-12, grad=4e-11, eeg=130, eog=150e-6, ecg=5e-4,
#  emg=1e2, ref_meg=1e-12, misc=1e-3, stim=1,
#  resp=1, chpi=1e-4, whitened=1e2))


### Subtract mean (1 pt) from every point in sliding window (raise/lower data) (reduce spatial bias/reduce 0 hz)
avgless_data = []

for w in sliding_window:
    w_data = w.get_data(picks=['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3'])
    # print (len(w_data), w_data.shape)
    w_avg = np.mean(w_data, axis=1)
    # print("AVERAGE SHAPE", w_avg.shape)
    w_data = (w_data.transpose() - w_avg.transpose()).transpose()
    # w_data -= w_avg
    avgless_data.append(mne.io.RawArray(w_data,info))


# avgless_data[0].plot(block = True, scalings=dict(mag=1e-12, grad=4e-11, eeg=130, eog=150e-6, ecg=5e-4,
#  emg=1e2, ref_meg=1e-12, misc=1e-3, stim=1,
#  resp=1, chpi=1e-4, whitened=1e2))


### Applying FFT
print(data.shape, len(data))
transformed_data = []
for channel in range(len(data)):
    window_data = []
    for window in avgless_data:
        window_data.append(np.fft.fft(window.get_data()[channel],n=window_size*2))
    transformed_data.append(np.asarray(window_data)) 
transformed_data = np.asarray(transformed_data)
print("TD SHAPE", transformed_data.shape)
transformed_data = np.abs(transformed_data)**2

### export transformed data as binary .npy
exported_data = transformed_data[:][:][:56]

t = time.localtime()
timestamp = time.strftime('%b-%d-%Y_%H%M', t)
file = ("FFT Data " + timestamp)
np.save(file,exported_data)

# x-label for frequencies.
N = window_size*2
freq = np.fft.fftfreq(N,d=1/sfreq)

print('N = ',N)


#channel to read psd from
channel = 1
# window = 14

psd = transformed_data[channel]

print (psd.shape)
psd = np.mean(psd, axis=0)
print (psd.shape)
# print (psd)

# print(np.mean(data[index]))
# psd -= np.abs(np.mean(data[index]))

print(psd[:56])
print(freq[:56])

# plot the power spectrum
# py.plot(psd2D)
plt.figure(1)
plt.clf()
# plt.xlim(0,120)
# plt.ylim(0,1e6)
plt.plot(freq[:56],psd[:56])
plt.show()

# plt.rcParams["figure.autolayout"] = True
# N = 256
# t = np.arange(N)
# m = 4
# nu = float(m)/N
# signal = np.sin(2*np.pi*nu*t)
# ft = np.fft.fft(signal)
# freq = np.fft.fftfreq(N)
# plt.plot(freq, ft.real**2 + ft.imag**2)
# plt.show() 