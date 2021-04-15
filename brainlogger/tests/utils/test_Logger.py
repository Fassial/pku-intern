"""
Created on 17:12, Apr. 15th, 2021
Author: fassial
Filename: test_Logger.py
"""
import os
import sys
sys.path.append("../..")
from brainlogger.utils import Logger

## macro
_LOGS = [
    bytes("Hello World", encoding = "utf-8"),
    bytes("hello world", encoding = "utf-8"),
    bytes("HELLO WORLD", encoding = "utf-8"),
]
DIR_OUTPUTS = os.path.join(os.getcwd(), "outputs")
if not os.path.exists(DIR_OUTPUTS): os.mkdir(DIR_OUTPUTS)
FILE_LOG = os.path.join(DIR_OUTPUTS, "test_Logger.log")

## def test func
# def test_Logger func
def test_Logger():
    # test start
    print("test_Logger...", end = "")

    # inst logger
    logger = Logger(fname = FILE_LOG)
    # write log
    for log in _LOGS:
        logger.write(log)
    # read logs
    logs = logger.read()
    # check log_r == log_w
    for i in range(len(_LOGS)):
        assert _LOGS[i] == logs[i]

    # test pass
    print("\rtest_Logger pass!")

## def main func
def main():
    # test Logger
    test_Logger()

if __name__ == "__main__":
    main()

