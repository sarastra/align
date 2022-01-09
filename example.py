import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from align import Align
from image import LuminanceCameraImage, SimulationImage

log = True
m = 'nove/D41 LH HV.pf'  # measurement image file
s = 'nove/HV.txt'  # simulation image file
c = 'nove/HV.camera'  # .camera file
alpha = 0.5

name = os.path.split(m)[1].split(sep='.')[0]

if log:
    log = 'log'
    if not os.path.exists(log):
        os.makedirs(log)
    filename = os.path.join(log, name + '.log')
    if os.path.exists(filename):
        os.remove(filename)
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[logging.FileHandler(filename),
                                  logging.StreamHandler()])

logging.info(m + ', ' + s + ', ' + c + '\n')

sim = SimulationImage(s, camera_filename=c)
logging.info('Initial camera:')
logging.info(sim.camera)

meas = LuminanceCameraImage(m)

align = Align(sim.im, meas.im)
M = align.find_M(alpha=alpha)

new_camera = sim.new_camera(M)
logging.info('New camera:')
logging.info(new_camera)
new_camera.save_camera()


# diagnostic images
if log:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.set_title('Initial difference between meas. and sim.')
    im = ax1.imshow(align.meas / np.max(align.meas) -
                    align.sim / np.max(align.sim),
                    vmin=-1, vmax=1,
                    cmap='bwr')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    ax2.set_title('Final difference between meas. and sim.')
    im = ax2.imshow(align.meas / np.max(align.meas) -
                    align.new_sim / np.max(align.new_sim),
                    vmin=-1, vmax=1,
                    cmap='bwr')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax)

    fig.tight_layout()
    fig.savefig(os.path.join(log, name + '_differences.png'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.set_title('Measurement spectrum')
    im = ax1.imshow(align.mfmeas_lp, cmap='jet')

    ax2.set_title('Simulation spectrum')
    im = ax2.imshow(align.mfsim_lp, cmap='jet')

    fig.tight_layout()
    fig.savefig(os.path.join(log, name + '_log_magnitude_log_polar.png'))
