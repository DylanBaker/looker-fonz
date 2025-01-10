#!/bin/bash

source venv/bin/activate

rm branch_comparison/results/*

git checkout master
python3 branch_comparison/generate_master_results.py

git checkout feature/async
python3 branch_comparison/generate_async_results.py

python3 branch_comparison/diff_results.py