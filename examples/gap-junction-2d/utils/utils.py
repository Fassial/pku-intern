"""
Created on 19:34, May. 23rd, 2021
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
def _get_omega(frate, N):
    # init omega
    omega = []

    # init idxs
    idxs = []
    for i in range(frate.shape[0]):
        idx = [(i + j) if (i + j) < frate.shape[0] else ((i + j) - frate.shape[0]) for j in range(N)]
        idxs.append(idx)

    # set omega
    for idx in idxs:
        frate_sub = frate[idx]
        omega.append(np.sum((frate_sub - np.mean(frate_sub)) ** 2) / N)

    return np.max(omega)

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
def get_omega(spike, dt):
    # init omega
    omega = []

    # init frate
    frate = np.sum(spike, axis = 1) / (spike.shape[1] * dt / 1000.)
    # set omega
    for i in range(spike.shape[0]):
        omega.append(_get_omega(
            frate = frate,
            N = i + 1
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

