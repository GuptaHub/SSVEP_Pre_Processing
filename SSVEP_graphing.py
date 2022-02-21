import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt


filename = 'OpenBCI-RAW-2021-11-15_Twelve.txt'
df = pd.read_csv(filename)
# df = df.loc[:, ' EXG Channel 0': ' EXG Channel 7']
df = df.loc[:, ' EXG Channel 0': ' EXG Channel 3']
print(df)


data = df.to_numpy()
data = np.transpose(data)
print(data)

# ch_names = ['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5',
#                 'EXG Channel 6', 'EXG Channel 7']

ch_names = ['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3']

sfreq = 250
info = mne.create_info(ch_names, sfreq, ch_types='eeg')

data = data.astype(float)
print(data)

raw = mne.io.RawArray(data[:,251:], info)
print(raw)
print(raw.info)

window_size = sfreq/2

sliding_window = []
temp = 250

# print("Length of Raw",len(raw[0][0][0]))
raw_len = len(raw[0][0][0]) + 250

print("Raw_len", raw_len)

# Implement Sliding Window
for sample in range(250,raw_len):
    if sample != 250 and sample % window_size == 0:
        # print("raw",raw[:])
        raw_temp = mne.io.RawArray(data[:,temp:sample],info)
        sliding_window.append(raw_temp)
        temp = sample
    # elif raw_len == sample-1:
    #     raw_temp = mne.io.RawArray(data[:,temp:sample],info)
    #     sliding_window.append(raw_temp)
    #     temp = sample
    #     break



print ("SLIDING WINDOW:",len(sliding_window))

for i in range(len(sliding_window)):
    print(len(sliding_window[i]))

# print ("TYPE RAW:", type(raw))
# print ("TYPE SLIDING:",type(sliding_window[0][0]))
# print ("SLIDING WINDOW [0]",sliding_window[0])


# raw.plot(block = True, scalings=dict(mag=1e-12, grad=4e-11, eeg=130, eog=150e-6, ecg=5e-4,
#  emg=1e2, ref_meg=1e-12, misc=1e-3, stim=1,
#  resp=1, chpi=1e-4, whitened=1e2))

# sliding_window[0].plot(block = True, scalings=dict(mag=1e-12, grad=4e-11, eeg=130, eog=150e-6, ecg=5e-4,
#  emg=1e2, ref_meg=1e-12, misc=1e-3, stim=1,
#  resp=1, chpi=1e-4, whitened=1e2))

# sliding_window[1].plot(block = True, scalings=dict(mag=1e-12, grad=4e-11, eeg=130, eog=150e-6, ecg=5e-4,
#  emg=1e2, ref_meg=1e-12, misc=1e-3, stim=1,
#  resp=1, chpi=1e-4, whitened=1e2))

# sliding_window[2].plot(block = True, scalings=dict(mag=1e-12, grad=4e-11, eeg=130, eog=150e-6, ecg=5e-4,
#  emg=1e2, ref_meg=1e-12, misc=1e-3, stim=1,
#  resp=1, chpi=1e-4, whitened=1e2))

###  Subtract average of all channels from each individual channel (per window) from input (raw) signal 
# print ("SLIDING WINDOW:",len(sliding_window))
# print ("SLIDING WINDOW [0]",type(sliding_window[0]))
# print ("Sliding window[0][0]:",type(sliding_window[0][0]))
# print ("Sliding window[0][0][0]:",sliding_window[0][0][0])
# print ("Sliding window[0][0][0][0]:",sliding_window[0][0][0][0])
# print ("Sliding window[0].get_data:",sliding_window[0].get_data(picks=['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3']))

avgless_data = []

for w in sliding_window:
    sliding_data = w.get_data(picks=['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3'])
    ch_average = []
    for ch in range(0,4):
        ch_average.append(np.mean(sliding_data[ch]))
    sliding_average = np.mean(ch_average)
    sliding_data -= sliding_average
    avgless_data.append(mne.io.RawArray(sliding_data,info))

# avgless_data[0].plot(block = True, scalings=dict(mag=1e-12, grad=4e-11, eeg=130, eog=150e-6, ecg=5e-4,
#  emg=1e2, ref_meg=1e-12, misc=1e-3, stim=1,
#  resp=1, chpi=1e-4, whitened=1e2))


### Applying FFT
print(data.shape)
transformed_data = []
# for channel in range(len(data)):
#     transformed_data.append(np.fft.fft(data[channel]))
# transformed_data = np.asarray(transformed_data)
for channel in range(len(data)):
    window_data = []
    for window in sliding_window:
        window_data.append(np.fft.fft(window.get_data()[channel]))
    transformed_data.append(np.asarray(window_data)) 
transformed_data = np.mean(np.asarray(transformed_data), axis=0)

print("TD:",transformed_data)


# Remove early data points (giant spike)
# new_transformed_data = []
# indices = [0,1,2,3,4,5]
# for channel in range (len(data)):
#     new_transformed_data.append(np.delete(transformed_data[channel],indices))
# new_transformed_data = np.asarray(new_transformed_data)
# print(new_transformed_data[0])
# N = data[0].size #-len(indices)
N = 125

# x-label for frequencies.
freq = np.fft.fftfreq(N,d=1/250)

print('N = ',N)

# transformed_data = mne.io.RawArray(transformed_data, info)
# transformed_data.plot(block = True, scalings=dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
#  emg=1e2, ref_meg=1e-12, misc=1e-3, stim=1,
#  resp=1, chpi=1e-4, whitened=1e2))

# x = np.random.random(1024)
# print(np.fft.fft(x))

### Graphing FFT magnitude
# thepower spectrum is:

#channel to read psd from
index = 1

psd = np.abs(transformed_data[index])
# print (psd)
# print(np.mean(data[index]))
# psd -= np.abs(np.mean(data[index]))

print(psd)
print(freq)

# plot the power spectrum
# py.plot(psd2D)
plt.figure(1)
plt.clf()
plt.xlim(-1,90)
plt.ylim(0,1e3)
plt.plot(freq,psd)
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