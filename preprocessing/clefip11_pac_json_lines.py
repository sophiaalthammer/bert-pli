import os
import argparse
import random
from statistics import mean
from collections import defaultdict
from operator import itemgetter
from itertools import groupby
import jsonlines
import re
import xml.etree.ElementTree as ET
random.seed(42)

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--train-dir', action='store', dest='train_dir',
                    help='train directory location', required=True)
parser.add_argument('--corpus-dir', action='store', dest='corpus_dir',
                    help='corpus directory', required=False)
parser.add_argument('--folder-name', action='store', dest='folder_name',
                    help='folder of bm25 retrieved documents', required=False)

args = parser.parse_args()

#train_dir = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTraining/'
#corpus_dir = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/corpus_small'
#folder_name = 'bm25_top50'

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

#
# only english topics!
#

with open(os.path.join(args.train_dir, 'english_topics.txt'), 'r') as f:
    english_topics = f.read().splitlines()

english_topics = [x.split('.xml')[0] for x in english_topics]


def read_in_patent_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    abstract_text = []
    # read in abstract text
    for abstract in root.iter('abstract'):
        if abstract.attrib['lang'] == 'EN':
            for text in abstract:
                try:
                    abstract_text.append(re.sub('\n', '',text.text))
                except:
                    pass

    # read in claims
    claims_text = {}
    for claims in root.iter('claims'):
        if claims.attrib['lang'] == 'EN':
            for claim in claims:
                for text in claim:
                    try:
                        claims_text.update({claim.attrib['num']: re.sub('\n', '',text.text)})
                    except:
                        pass

    # read in descriptions
    desc_text = {}
    for descriptions in root.iter('description'):
        if descriptions.attrib['lang'] == 'EN':
            for desc in descriptions:
                try:
                    desc_text.update({desc.attrib['num']: re.sub('\n', '',desc.text)})
                except:
                    pass

    text = [x for x in abstract_text if x is not None]
    text.extend([x for x in claims_text.values() if x is not None])
    text.extend([x for x in desc_text.values() if x is not None])
    return text


with jsonlines.open(os.path.join(args.train_dir, 'train_org.json'), mode='w') as writer:
    for topic in gold_labels.keys():
        if topic in english_topics:
            try:
                #topic = 'EP-1229507-A2'
                labels = gold_labels.get(topic)
                bm25 = bm25_top50.get(topic)



                bm25_negatives = random.sample(list(set(bm25)-set(labels)), 50-len(labels))
                candidates = labels + bm25_negatives

                # öffne die dateien aus den files, also die topics
                topic_text = read_in_patent_xml(os.path.join(args.train_dir, 'files', '{}.xml'.format(topic)))

                for candidate in candidates:
                    #candidate = 'WO-1979000001'
                    candidate_name = candidate.split('-')
                    if candidate_name[0] == 'EP':
                        file_path = os.path.join(args.corpus_dir, candidate_name[0], '00000{}'.format(candidate_name[1][0]), candidate_name[1][1:3], candidate_name[1][3:5], candidate_name[1][5:7])
                    else:
                        file_path = os.path.join(args.corpus_dir, candidate_name[0], '00{}'.format(candidate_name[1][0:4]),
                                             candidate_name[1][4:6], candidate_name[1][6:8], candidate_name[1][8:10])
                    # öffne die dateien aus dem corpus dir, mit xml als endung, immer das erste (weil ich die endung von a1 a2 nicht mitspeichere)
                    candidate_text = read_in_patent_xml(os.path.join(file_path, os.listdir(file_path)[0]))
                    if len(candidate_text) > 4:
                        writer.write({'guid': '{}_{}'.format(topic, candidate),
                                      'q_paras': topic_text,
                                      'c_paras': candidate_text,
                                      'label': 1 if candidate in labels else 0})
                    else:
                        print('This topic {} and this candidate {} dont have a long enough text (longer than 4)'.format(topic, candidate))
            except:
                print('for this topic it didnt work {}'.format(topic))

