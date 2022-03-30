import os
import argparse
import random
import jsonlines
import json
import re
import random
random.seed(412)

parser = argparse.ArgumentParser()

parser.add_argument('--output_run', action='store', dest='output_run',
                    help='training file directory location', required=True)
parser.add_argument('--run_list', action='store', dest='run_list',
                    help='training file directory location', required=True)
parser.add_argument('--shuffle', action='store', dest='shuffle',
                    help='training file directory location', required=False)
args = parser.parse_args()

run_merged = args.output_run
run_list = args.run_list
run_list = run_list.split(',')

shuffle = args.shuffle

if not shuffle:
    with jsonlines.open(run_merged, mode='w') as writer:
        for run in run_list:
            with open(run, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    writer.write(sample)
else:
    with jsonlines.open(run_merged, mode='w') as writer:
        samples = []
        for run in run_list:
            with open(run, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    samples.append(sample)

        random.shuffle(samples)
        for sample in samples:
            writer.write(sample)
