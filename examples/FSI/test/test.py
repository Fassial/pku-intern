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
import test_utils

def main():
    ## test_stimulus
    # test_inputs.test_poisson_input
    # test_stimulus.test_inputs.test_poisson_input()

    # test_stimulus
    # test_stimulus.test_stimulus()
    # test_stimulus_normal
    # test_stimulus.test_stimulus_normal()
    # test_stimulus_frate_increase
    test_stimulus.test_stimulus_frate_increase()

    ## test_utils
    # test_get_omega
    # test_utils.test_get_omega()

    # test_get_cv
    # test_utils.test_get_cv()

if __name__ == "__main__":
    main()

