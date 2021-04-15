"""
Created on 15:06, Apr. 15th, 2021
Author: fassial
Filename: Logger.py
Description:
    Read & Write record. The code was modified from https://github.com/dmlc/tensorboard
"""
import re
import struct

from .crc32c import crc32c

__all__ = [
    "Logger",
]

## def Logger class
class Logger(object):
    # init macro
    _struct_len = {
        "I": 4,
        "Q": 8,
    }

    def __init__(self, fname):
        # init params
        self.fname = fname

        # init vars
        self._writer = open(self.fname, "wb")
        self._reader = open(self.fname, "rb")

    def __del__(self):
        # close writer
        if self._writer is not None:
            self._writer.close()
        # close reader
        if self._reader is not None:
            self._reader.close()

    def write(self, log):
        # write len-crc(len)-log-crc(log)
        header = struct.pack("Q", len(log))
        self._writer.write(header)
        self._writer.write(struct.pack("I", _masked_crc32c(header)))
        self._writer.write(log)
        self._writer.write(struct.pack("I", _masked_crc32c(log)))
        # flush writer
        self._writer.flush()

    def read(self):
        # init logs
        logs = []
        # get header
        header = self._reader.read(Logger._struct_len["Q"])
        while header:
            # unpack header
            log_len = struct.unpack("Q", header)[0]
            # get header_crc & check crc
            header_crc = struct.unpack("I", self._reader.read(Logger._struct_len["I"]))[0]
            if header_crc != _masked_crc32c(header):
                raise ValueError("Failed at checking crc of header in utils.Logger.")
            # get log
            log = self._reader.read(log_len)
            # get log_crc & check crc
            log_crc = struct.unpack("I", self._reader.read(Logger._struct_len["I"]))[0]
            if log_crc != _masked_crc32c(log):
                raise ValueError("Failed at checking crc of log in utils.Logger.")

            # add log to logs
            logs.append(log)

            # update header
            header = self._reader.read(Logger._struct_len["Q"])

        # reset frp
        self._reader.seek(0, 0)

        return logs

## def helper func
# def _u32 func
def _u32(x):
    return x & 0xffffffff

# def _masked_crc32c func
def _masked_crc32c(data):
    x = _u32(crc32c(data))
    return _u32(((x >> 15) | _u32(x << 17)) + 0xa282ead8)

