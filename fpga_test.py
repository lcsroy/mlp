from pynq import Overlay
import pynq.lib.dma
from pynq import allocate
import numpy as np
INPUT_NODES = 33
OUTPUT_NODES = 8

def fpga_mlp(extracted_data):
    overlay = Overlay("xilinx@pynq:/home/mlp/mlp_wrapper.bit")
    dma = overlay.axi_dma_0

    in_buffer = allocate(shape = (INPUT_NODES,), dtype = np.int32)
    out_buffer = allocate(shape = (OUTPUT_NODES,), dtype = np.int32)
    if (len(extracted_data) != 33):
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

extracted_data = [1.355245,1.238677,0.866767,1.297441,0.230306,-1.343787,0.817745,1.015048,1.825221,-0.682043,1.504707,-0.829518,-0.496181,-0.366399,-0.837389,0.376625,-0.693459,-0.666634,-0.800256,-0.583502,1.199955,-0.332868,-0.382635,-0.392527,-0.425029,-0.428253,1.297535,0.572801,-0.358504,-0.255945,-0.648057,-0.259146,-0.768899]
fpga_mlp(extracted_data)