"""
Created on 12:01, May. 23rd, 2021
Author: fassial
Filename: test.py
"""
import brainpy as bp
# local dep
import sys
sys.path.append("..")
import test_stimulus

def main():
    ## test_stimulus
    # test_stimulus.test_inputs
    # test_stimulus.test_inputs.test_poisson_input()

    # test_stimulus
    test_stimulus.test_stimulus()

if __name__ == "__main__":
    main()

