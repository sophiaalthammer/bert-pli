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
                    help='training file directory location', required=False, default=False)
parser.add_argument('--balanced', action='store', dest='balanced',
                    help='training file directory location', required=False, default=False)
args = parser.parse_args()

run_merged = args.output_run
run_list = args.run_list
run_list = run_list.split(',')

shuffle = args.shuffle
balanced = args.balanced


if shuffle:
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
elif balanced:
    with jsonlines.open(run_merged, mode='w') as writer:
        samples = []
        for run in run_list:
            guids = []
            with open(run, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    samples.append(sample)

            all_docs = {}
            query_run = {}
            for sample in samples:
                all_docs.update({sample.get('guid'): {'label': sample.get('label'), 'res': sample.get('res')}})
                query_id = sample.get('guid').split('_')[0]
                if query_run.get(query_id):
                    query_run.get(query_id).update({sample.get('guid'): sample.get('label')})
                else:
                    query_run.update({query_id: {}})
                    query_run.get(query_id).update({sample.get('guid'): sample.get('label')})

            for query_id, values in query_run.items():
                pos_guids = []
                neg_guids = []
                for guid, label in values.items():
                    if label == 1:
                        pos_guids.append(guid)
                    else:
                        neg_guids.append(guid)

                if len(neg_guids) <= len(pos_guids):
                    pos_guids.extend(neg_guids)
                else:
                    neg_guids_balanced = random.sample(neg_guids, len(pos_guids))
                    pos_guids.extend(neg_guids_balanced)

                guids.extend(pos_guids)

            for guid in guids:
                writer.write(
                    {'guid': guid, 'res': all_docs.get(guid).get('res'), 'label': all_docs.get(guid).get('label')})

else:
    with jsonlines.open(run_merged, mode='w') as writer:
        for run in run_list:
            with open(run, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    writer.write(sample)


run = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2022/task1/bertpli/poolout_output/train/run_merged_withpos_shuffled_balanced.json'

with open(run, 'r') as f:
    for line in f:
        sample = json.loads(line)
        samples.append(sample)

all_docs = {}
query_run = {}
for sample in samples:
    all_docs.update({sample.get('guid'): {'label': sample.get('label'), 'res': sample.get('res')}})
    query_id = sample.get('guid').split('_')[0]
    if query_run.get(query_id):
        query_run.get(query_id).update({sample.get('guid'): sample.get('label')})
    else:
        query_run.update({query_id: {}})
        query_run.get(query_id).update({sample.get('guid'): sample.get('label')})

