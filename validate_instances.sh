#!/bin/bash
rm -rf logs
python codearena.py --BugFixing --predictions_path gold --instance_ids apache__airflow-46883 --run_id test