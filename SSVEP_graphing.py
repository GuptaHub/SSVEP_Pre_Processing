import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt


filename = 'OpenBCI-RAW-2021-11-15_Twelve.txt'
df = pd.read_csv(filename)
df = df.loc[:, ' EXG Channel 0': ' EXG Channel 7']
print(df)


data = df.to_numpy()
data = np.transpose(data)
print(data)

ch_names = ['EXG Channel 0', 'EXG Channel 1', 'EXG Channel 2', 'EXG Channel 3', 'EXG Channel 4', 'EXG Channel 5',
                'EXG Channel 6', 'EXG Channel 7']

sfreq = 250
info = mne.create_info(ch_names, sfreq, ch_types='eeg')

data = data.astype(float)


raw = mne.io.RawArray(data, info)
print(raw)
print(raw.info)

# raw.plot(block = True, scalings=dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
#  emg=1e2, ref_meg=1e-12, misc=1e-3, stim=1,
#  resp=1, chpi=1e-4, whitened=1e2))

### Applying FFT
print(data.shape)
transformed_data = []
for channel in range(len(data)):
    transformed_data.append(np.fft.fft(data[channel]))
transformed_data = np.asarray(transformed_data)

# ## Remove early data points (giant spike)
new_transformed_data = []
indices = [0,1,2,3,4,5]
for channel in range (len(data)):
    new_transformed_data.append(np.delete(transformed_data[channel],indices))
new_transformed_data = np.asarray(new_transformed_data)
# print(new_transformed_data[0])
N = len(data[0])-len(indices)
freq = np.fft.fftfreq(N)

print('N = ',N)



# transformed_data = mne.io.RawArray(transformed_data, info)
# transformed_data.plot(block = True, scalings=dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
#  emg=1e2, ref_meg=1e-12, misc=1e-3, stim=1,
#  resp=1, chpi=1e-4, whitened=1e2))

# x = np.random.random(1024)
# print(np.fft.fft(x))

### Graphing FFT magnitude
# thepower spectrum is:
psd = np.abs(new_transformed_data[3])

print(psd)
print(freq)

# plot the power spectrum
# py.plot(psd2D)
plt.figure(1)
plt.clf()
plt.ylim(-1e4,1e5)
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