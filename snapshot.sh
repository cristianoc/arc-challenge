#!/bin/bash

./test.sh
python src/simple.py
python src/predict_size.py
python src/predict_colors.py
python src/ablation_size.py
python src/ablation_colors.py

