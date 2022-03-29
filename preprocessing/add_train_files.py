import os
import argparse
import random
import jsonlines
import json
import re

parser = argparse.ArgumentParser()

parser.add_argument('--output_run', action='store', dest='output_run',
                    help='training file directory location', required=True)
parser.add_argument('--run_list', action='store', dest='run_list',
                    help='training file directory location', required=True)
args = parser.parse_args()

run_merged = args.output_run
run_list = args.run_list
run_list = run_list.split(',')

with jsonlines.open(run_merged, mode='w') as writer:
    for run in run_list:
        with open(run, 'r') as f:
            for line in f:
                sample = json.loads(line)
                writer.write(sample)
