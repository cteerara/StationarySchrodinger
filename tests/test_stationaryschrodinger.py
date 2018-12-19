#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `stationaryschrodinger` package."""

import pytest
import math
import sys
import os
import tensorflow as tf
tf.enable_eager_execution();

sys.path.append("../")
from click.testing import CliRunner

from stationaryschrodinger import cli
from stationaryschrodinger import tfAPI
from stationaryschrodinger import ssIO
from stationaryschrodinger import EigFunc

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
tol = 1e-6
def test_compare():
  t1 = tf.constant([1],dtype=tf.float64)
  t2 = tf.constant([[1],[2]],dtype=tf.float64)
  t3 = tf.constant([[1,1],[2,2]],dtype=tf.float64)
  t4 = tf.constant([[1,1],[2,3]],dtype=tf.float64)
  # Case: dimension is not equal
  assert(not tfAPI.compare(t1,t2,tol))
  # Case: type mismatch
  assert(not tfAPI.compare(t1,tf.cast(t1,tf.float32),tol))
  # Case: size noe equal
  assert(not tfAPI.compare(t2,t3,tol))
  # Case: elements not equal
  assert(not tfAPI.compare(t3,t4,tol))
  # case: IS equal
  assert(tfAPI.compare(t3,t3+1e-7,tol)) # Check if it exceeds tol

def test_tfdim():
  t1 = tf.constant(0,shape=[2,2])
  tdim = tf.constant(2)
  assert(tfAPI.compare(tfAPI.tfdim(t1),tdim,tol))

def test_tflen():
  t1 = tf.constant([1,2,3])
  assert(tfAPI.compare( tfAPI.tflen(t1),tf.constant(3),tol))

def test_integrate():
  x = tf.Variable([2,4,6])
  F = tf.Variable([1,1,1])
  Fint = tf.Variable(4)
  assert(tfAPI.compare( Fint, tfAPI.integrate(F,F,x),tol))
  
#-------------------------------------------------------------------#
#------------- TEST IO.py ------------------------------------------#
#-------------------------------------------------------------------#

def test_IO_readPotentialEnergy():
  tol = 1e-6
  x = ssIO.readPotentialEnergy('tests/testIO_pot.dat')
  assert(tfAPI.compare(x[0],tf.constant([0,1,2],dtype=tf.float64),tol))
  assert(tfAPI.compare(x[1],tf.constant([0,1,2],dtype=tf.float64),tol))

def test_IO_readConsts():
  x = ssIO.readConsts('tests/testIO_consts.dat')
  assert(x[0] == 1)
  assert(x[1] == 3)

#-------------------------------------------------------------------#
#------------- TEST EigFunc.py -------------------------------------#
#-------------------------------------------------------------------#

def test_FPoly():
  pi = 4*math.atan(1.0)
  x = [0,pi/2,pi]
  x = tf.constant(x,dtype=tf.float32)
  b = tf.cast(EigFunc.FPoly(x,3),tf.float32)
  bActual = [tf.constant([1,1,1],dtype=tf.float32),tf.constant([0,1,0],dtype=tf.float32),tf.constant([1,0,-1],dtype=tf.float32)]
  for i in range(0,3):  
    assert(tfAPI.compare( b[i] , bActual[i],tol))

def test_project():
  pi = 4*math.atan(1.0)
  x = [0,pi/2,pi]
  x = tf.constant(x,dtype=tf.float32)
  b = EigFunc.FPoly(x,3)
  F = tf.Variable([0,1,0],dtype=tf.float32)
  FProjActual = tf.constant([0,1,0],dtype=tf.float32)
  FProj = EigFunc.project(F,b,x)
  assert(tfAPI.compare(FProj,FProjActual,tol))

def test_Hij():
  pi = 4*math.atan(1.0)
  n = 100
  x = []
  F = []
  c = 1
  for i in range(0,n+1):
    x.append(i*2*pi/n)
    F.append(1.0)
  x = tf.constant(x,dtype=tf.float32)
  F = tf.constant(F,dtype=tf.float32)
  b = EigFunc.FPoly(x,3)
  Hij = EigFunc.hamil(x,b,c,F)
  
  # Construct Hij Actual
  HijActual = [[0,0,0],[0,0,0],[0,0,0]]
  
  HijActual[0][0] = 2*pi
  HijActual[1][1] = c+pi
  HijActual[2][2] = c+pi  

  HijActual = tf.constant(HijActual,dtype=tf.float32)

  assert(tfAPI.compare(Hij,HijActual,tol))

def test_Eig():
  tol = 1e-6
  LES = EigFunc.LowestEnergyState(tf.constant([[2.,0.],[0.,1.]]))
  assert( tfAPI.compare( tf.constant(1.),LES[0],tol ) )
  assert( tfAPI.compare( tf.constant([0.,1.]), LES[1],tol ) )


