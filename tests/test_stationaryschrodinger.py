#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `stationaryschrodinger` package."""

import pytest
import sys
import os
import tensorflow as tf
tf.enable_eager_execution();

sys.path.append("../")
from click.testing import CliRunner

from stationaryschrodinger import stationaryschrodinger
from stationaryschrodinger import cli
from stationaryschrodinger import tfAPI
from stationaryschrodinger import IO

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'stationaryschrodinger.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output

#-------------------------------------------------------------------#
#------------- TEST tfAPI.py ---------------------------------------#
#-------------------------------------------------------------------#

def test_tfAPI_compare():
  t1 = tf.constant([1],dtype=tf.float64)
  t2 = tf.constant([[1],[2]],dtype=tf.float64)
  t3 = tf.constant([[1,1],[2,2]],dtype=tf.float64)
  t4 = tf.constant([[1,1],[2,3]],dtype=tf.float64)
  # Case: dimension is not equal
  assert(not tfAPI.compare(t1,t2))
  # Case: size noe equal
  assert(not tfAPI.compare(t2,t3))
  # Case: elements not equal
  assert(not tfAPI.compare(t3,t4))
  # case: IS equal
  assert(tfAPI.compare(t3,t3))

def test_tfAPI_zeroes():
  t1 = tfAPI.zeroes([2,2])
  t2 = tf.Variable([[0.,0.],[0.,0.]],dtype=tf.float64)
  assert(tfAPI.compare(t1,t2))

#-------------------------------------------------------------------#
#------------- TEST IO.py ------------------------------------------#
#-------------------------------------------------------------------#

def test_IO_readPotentialEnergy():
  x = IO.readPotentialEnergy('tests/IOtest.dat')
  assert(tfAPI.compare(x[0],x[0])
