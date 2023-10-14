import numpy as np
from numpy import mean, std, sqrt, percentile, argmax
from scipy.stats import skew, kurtosis, mode
from sklearn.preprocessing import StandardScaler

x_window = []
y_window = []
z_window = []
input_features = []
WINDOW_SIZE = 5
scaler = StandardScaler()

def data_processing(data):
    global x_window, y_window, z_window, input_features
    features = []
    x_features = []
    y_features = []
    z_features = []

    x_window.append(data[0])
    y_window.append(data[1])
    z_window.append(data[2])
    
    #print(x_window)
    if len(input_features) == 33:
        input_features = []
        features = []
    if len(x_window) == WINDOW_SIZE:
        x_features = feature_extraction(x_window)
      #  print(x_features)
        features = features + x_features
       # print(features)
        x_features = []
        x_window = []
    if len(y_window) == WINDOW_SIZE:
        y_features = feature_extraction(y_window)
      #  print(y_features)
        features = features + y_features
       # print(features)
        y_features = []
        y_window = []
    if len(z_window) == WINDOW_SIZE:
        z_features = feature_extraction(z_window)
        #print(z_features)
        features = features + z_features
        print(input_features)
        z_features = []
        z_window = []
        features = np.array(features).reshape(-1, 1)
        features = scaler.fit_transform(features)
        features = np.hstack(features)
    input_features = features
    print(input_features)
    return input_features

def feature_extraction(window):
    extracted_data = []

    #time-domain features
    nmean = np.mean(window)
    nstd = np.std(window)
    nvar = np.var(window)
    nmedian = np.median(window)
    nskew = skew(window)
    nkurt = kurtosis(window)
    nptp = np.ptp(window)
    nmode = mode(window, axis = None, keepdims = False)[0]

    #frequency-domain features
    fft_window = np.fft.rfft(window)

    fmax = abs(max(fft_window))
    fmin = abs(min(fft_window))
    fen = sum(abs(fft_window) ** 2) / 100 ** 2

    
    extracted_data.append(nmean)
    extracted_data.append(nstd)
    extracted_data.append(nvar)
    extracted_data.append(nmedian)
    extracted_data.append(nskew)
    extracted_data.append(nkurt)
    extracted_data.append(nptp)
    extracted_data.append(nmode)
    extracted_data.append(fmax)
    extracted_data.append(fmin)
    extracted_data.append(fen)

    return extracted_data

data_processing([-4.71, -1.1, -0.99])
data_processing([2.14, -1.33, -1.31])
data_processing([2.42, -1.49, -1.51])
data_processing([2.63, -1.6, -1.66])
data_processing([2.79, -1.7, -1.77])
data_processing([2.94, -1.79, -1.88])
data_processing([-2.71, -1.1, -1.9])
data_processing([-3.71, -1.2, -1.59])
data_processing([-4.71, -1.3, -1.9])
data_processing([-4.71, -1.3, -1.9])


