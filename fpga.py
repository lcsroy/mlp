from pynq import Overlay
import pynq.lib.dma
from pynq import allocate
import numpy as np
from numpy import mean, std, sqrt, percentile, argmax
from scipy.stats import skew, kurtosis, mode
from sklearn.preprocessing import StandardScaler

INPUT_NODES = 33
OUTPUT_NODES = 8

RELOAD = 0
GRENADE = 1
FIST = 2
HAMMER = 3
PORTAL = 4
SPEAR = 5
SPIDERWEB = 6
SHIELD = 7

x_window = []
y_window = []
z_window = []
input_features = []
WINDOW_SIZE = 100
scaler = StandardScaler()


def fpga_mlp(extracted_data):
    overlay = Overlay(bitstream filepath)
    dma = overlay.axi_dma_0

    in_buffer = allocate(shape = (INPUT_NODES,), dtype = np.int32)
    out_buffer = allocate(shape = (OUTPUT_NODES,), dtype = np.int32)
    if (len(input_features) != 33):
        print("Do not have 33 input features!")
    else:
        for i in range(len(extracted_data)):
            in_buffer[i] = extracted_data[i]
    dma.sendchannel.transfer(in_buffer)
    dma.recvchannel.transfer(out_buffer)
    dma.sendchannel.wait()
    dma.recvchannel.wait()

    action_num = np.argmax(out_buffer)
    print(action_num)
    return action_num

def data_processing(data):
    global x_window, y_window, z_window, input_features
    #to store all the features after successful feature extraction
    features = []
    #store respective extracted features
    x_features = []
    y_features = []
    z_features = []
    #Append the data to respective windows and wait for 100 data to do feature extraction
    x_window.append(data[0])
    y_window.append(data[1])
    z_window.append(data[2])
    #Clear feature list to store new sets of data
    if len(input_features) == INPUT_NODES:
        input_features = []
        features = []
    #Feature extraction done for WINDOW_SIZE in x direction
    if len(x_window) == WINDOW_SIZE:
        x_features = feature_extraction(x_window)
        features = features + x_features
        x_features = []
        x_window = []
    #Feature extraction done for WINDOW_SIZE in y direction
    if len(y_window) == WINDOW_SIZE:
        y_features = feature_extraction(y_window)
        features = features + y_features
        y_features = []
        y_window = []
    #Feature extraction done for WINDOW_SIZE in z direction    
    if len(z_window) == WINDOW_SIZE:
        z_features = feature_extraction(z_window)
        features = features + z_features
        z_features = []
        z_window = []
        #Standardize the features
        features = np.array(features).reshape(-1, 1)
        features = scaler.fit_transform(features)
        features = np.hstack(features)

    input_features = features
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


