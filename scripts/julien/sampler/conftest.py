#! /usr/bin/python3.7
# -*- coding: utf-8 -*-

""""Test configuration file for pytest."""

import pytest


def pytest_addoption(parser):
    parser.addoption("--csv", action="store", required=True, type=str,
                     help="Absolut or relative (to execution directory) path to CSV file.")
    parser.addoption("--json", action="store", required=True, type=str,
                     help="Absolut or relative (to execution directory) path to JSON file.")
    parser.addoption("--threshold", action="store", type=float, default=0.1,
                     help="Relative threshold for calculated value to match expeceted value. Default is 0.01.")


# This is called for every test. Only get/set command line argument
def pytest_generate_tests(metafunc):
    threshold = metafunc.config.option.threshold
    if (threshold is not None):
        metafunc.parametrize("threshold", [threshold])


# Define command line arguments as part of pytest (conftest. has to bee in root dir to find paths)
def pytest_configure(config):
    pytest.csv_path = config.getoption("--csv")
    pytest.json_path = config.getoption("--json")
