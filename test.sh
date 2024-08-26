#!/bin/bash

DISPLAY=False pytest -s --log-level=INFO src/*.py

echo "All tests completed."
