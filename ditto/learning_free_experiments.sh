#!/bin/bash
source ~/.bashrc
echo "Running static_reimplementation.py"
python static_reimplementation.py
echo "Running last_reimplementation.py"
python last_reimplementation.py
echo "Running first_last_reimplementation.py"
python first_last_reimplementation.py
