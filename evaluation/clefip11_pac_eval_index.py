import os
import argparse
import random
from statistics import mean
from operator import itemgetter
from itertools import groupby
random.seed(42)

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--train-dir', action='store', dest='train_dir',
                    help='train directory location', required=True)
parser.add_argument('--folder-name', action='store', dest='folder_name',
                    help='folder name of the retrieved documents', required=False)


args = parser.parse_args()

#train_dir = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTraining/'
#folder_name = 'bm25_top20_250'

#
# load directory structure
#

list_dir = [x for x in os.walk(os.path.join(args.train_dir, args.folder_name))]


#
# read in top 50 scores per document
#

bm25_top50 = {}

for file in list_dir[0][2]:
    with open(os.path.join(args.train_dir, args.folder_name, file), 'r') as f:
        bm25_top50.update({file.split('_')[-1].split('.xml.txt')[0]: ['-'.join(x.split('-')[0:2]) for x in [x.strip() for x in f.read().splitlines()]]})

print(len(bm25_top50))

#
# read in the gold labels
#

with open(os.path.join(args.train_dir, 'qrels.txt'), 'r') as f:
    qrels = f.read().splitlines()

labels = {}
tuple_list = []
for qrel in qrels:
    qrel_splitted = qrel.split('\t')
    tuple_list.append((qrel_splitted[0], qrel_splitted[1]))

gold_labels = dict((k, [v[1] for v in itr]) for k, itr in groupby(tuple_list, itemgetter(0)))

print(len(gold_labels))

#
# only english topics!
#

with open(os.path.join(args.train_dir, 'english_topics_val.txt'), 'r') as f:
    english_topics = f.read().splitlines()

english_topics = [x.split('.xml')[0] for x in english_topics]


recall = []
for topic in gold_labels.keys():
    if topic in english_topics:
        labels = gold_labels.get(topic)
        bm25 = bm25_top50.get(topic)

        recall.append(len(set(labels) & set(bm25)) / len(labels))

print(mean(recall))

