import os
import argparse
import random
import jsonlines
import json
import re
random.seed(42)

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--run_before_poolout', action='store', dest='run_before_poolout',
                    help='training file directory location', required=True)
parser.add_argument('--run_after_poolout', action='store', dest='run_after_poolout',
                    help='training file directory location', required=True)
args = parser.parse_args()

#run_before_poolout = '/mnt/c/Users/sophi/Documents/phd/data/coliee2022/task1/train/run_for_bertpli_top1_15_0.json'
#run_after_poolout = '/mnt/c/Users/sophi/Documents/phd/data/coliee2022/task1/bertpli/output/train/run_for_bertpli_top1_15_0.json'

run_before_poolout = args.run_before_poolout
run_after_poolout = args.run_after_poolout

# open the train org document and create a match of guid and the label
# this is the file before the poolout, mine is run_for_bertpli_top1_15_0.json etc
# only inference for validation and test!
labels_list = []
with open(run_before_poolout) as f:
    for line in f:
        labels_list.append(json.loads(line))

match = {}
for label in labels_list:
    match.update({label.get('guid'): label.get('label')})

# assign guid and labels from that file
# augment the pooled out file with the labels! train_poolout is the pooled out filed
# actually i dont really know if i need that step if i dont need the labels, but lets see!
# also for test i only have the label 0 whatever

with jsonlines.open(run_after_poolout+'withlabels.json', mode='w') as writer:
    with open(run_after_poolout, 'r') as f:
        for line in f:
            sample = json.loads(line)
            guid = sample.get('guid')

            writer.write({'guid': guid,
                          'res': sample.get('res'),
                          'label': match.get(guid)})
