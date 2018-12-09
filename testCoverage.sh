#!/bin/bash
coverage run --source=stationaryschrodinger setup.py test
coverage report -m
