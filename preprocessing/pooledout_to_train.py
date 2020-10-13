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

parser.add_argument('--train-dir', action='store', dest='train_dir',
                    help='training file directory location', required=True)

args = parser.parse_args()

#train_dir = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_train/train.json'
#train_org_dir = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_train/train_org.json'


# open the train org document and create a match of guid and the label

labels_list = []
with open(os.path.join(args.train_dir, 'train_org.json')) as f:
    for line in f:
        labels_list.append(json.loads(line))

match = {}
for label in labels_list:
    match.update({label.get('guid'): label.get('label')})


with jsonlines.open(os.path.join(args.train_dir, 'train_poolout_totrain.json'), mode='w') as writer:
    with open(os.path.join(args.train_dir, 'train_poolout.json'), 'r') as f:
        for line in f:
            sample = json.loads(line)
            guid = sample.get('guid')

            writer.write({'guid': guid,
                          'res': sample.get('res'),
                          'label': match.get(guid)})


