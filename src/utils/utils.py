"""
Created on 19:46, June. 5th, 2021
Author: fassial
Filename: utils.py
"""
import numpy as np

__all__ = [
    "get_omega",
    "get_cv",
]

## define helper func
# define _get_omega func
def _get_omega(frate, bn):
    # init omega
    omega = []

    # init idxs
    idxs = []
    for i in range(int(np.ceil(frate.shape[0] / bn))):
        idx = []
        for j in range(bn):
            if (i * bn + j) < frate.shape[0]: idx.append(i * bn + j)
        idxs.append(idx)

    # set frate_bn
    frate_bn = np.array([np.mean(frate[idx,:], axis = 0) for idx in idxs])
    # set omega
    omega = np.sum((frate_bn - np.mean(frate_bn)) ** 2) / frate_bn.size

    return omega

# define _spike2isi func
def _spike2isi(spike, dt):
    # init isi
    isi = []

    # set isi
    spike_time = np.where(spike == 1)[0].astype(dtype = np.float32)
    for i in range(len(spike_time) - 1):
        isi.append(spike_time[i+1] - spike_time[i])
    isi = np.array(isi, dtype = np.float32); isi *= (dt / 1000.)

    return isi

# define _get_cv func
def _get_cv(spike, dt):
    # init cv
    cv = 0

    if (np.sum(spike) > 1):
        # get isi
        isi = _spike2isi(
            spike = spike,
            dt = dt
        )
        cv = np.sqrt(np.sum((isi - np.mean(isi)) ** 2) / isi.shape[0]) / np.mean(isi)

    return cv

## define utils func
# define get_omega func
def get_omega(spike, bin, dt, N = 100):
    # check spike.shape[1] // (bin / dt)
    assert int(spike.shape[1] / (bin / dt)) == spike.shape[1] / (bin / dt)

    # init omega
    omega = []

    # init frate
    frate = np.sum(spike.reshape(
        (spike.shape[0], int(spike.shape[1] / (bin / dt)), int(bin / dt))
    ), axis = 2) / (bin / 1000.)
    # set omega
    for i in range(N):
        omega.append(_get_omega(
            frate = frate,
            bn = i + 1
        ))

    return np.sqrt(np.max(omega))

# define get_cv func
def get_cv(spike, dt):
    # init cv
    cv = []

    # set cv
    for i in range(spike.shape[0]):
        cv.append(_get_cv(
            spike = spike[i,:],
            dt = dt
        ))

    return np.array(cv, dtype = np.float32)

