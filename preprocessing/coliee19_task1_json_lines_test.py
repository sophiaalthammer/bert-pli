import os
import csv
import argparse
import jsonlines
import re
import xml.etree.ElementTree as ET
import random
random.seed(42)

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--train-dir', action='store', dest='train_dir',
                    help='training file directory location', required=True)

parser.add_argument('--output-dir', action='store', dest='output_dir',
                    help='training output file location for test.tsv', required=True)

parser.add_argument('--test-gold-labels', action='store', dest='test_gold_labels',
                    help='location and name of the gold labels xml-file to create a training set from the test data', required=True)

args = parser.parse_args()

#train_dir = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test'
#output_dir = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1'
#test_gold_labels = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test_golden-labels.xml'

#
# load directory structure
#

list_dir = [x for x in os.walk(args.train_dir)]

#
# load gold labels as dictionary
#

tree = ET.parse(args.test_gold_labels)
root = tree.getroot()

gold_labels = {}
for child in root:
    rank = child.find('cases_noticed').text
    rank = rank.split(',')
    gold_labels.update({child.attrib['id']: rank})

#
# Write test.tsv file with query_id \t doc_id \t query_text \t doc_relevant_text
#

list_dir = [x for x in os.walk(args.train_dir)]

with jsonlines.open(os.path.join(args.train_dir, 'test_org.json'), mode='w') as writer:
    for sub_dir in list_dir[0][1]:
    #sub_dir = '001'
        with open(os.path.join(args.train_dir, sub_dir, 'base_case.txt'), 'r') as entailed_fragment:
            query_text = re.sub(r'\n+', '\n', re.sub(r'\n ', '\n', entailed_fragment.read()).strip()).splitlines()

        # read in bm25 top 50
        with open(os.path.join(args.train_dir, sub_dir, 'bm25_top50.txt'), 'r') as entailing_para:
            bm25_top50 = entailing_para.read().splitlines()
            bm25_top50 = [doc.strip() for doc in bm25_top50]

        doc_rel_id = gold_labels.get(sub_dir)

        # read in all paragraphs with their names and then choose the relevant ones and sample irrelevant ones!
        list_sub_dir_paragraphs = [x for x in os.walk(os.path.join(args.train_dir, sub_dir, 'candidates'))]
        paragraphs_text = {}
        for paragraph in list_sub_dir_paragraphs[0][2]:
            guid = '{}_{}'.format(sub_dir, paragraph.split('.')[0])
            if guid in bm25_top50:
                with open(os.path.join(args.train_dir, sub_dir, 'candidates', paragraph), 'r') as paragraph_file:
                    para_text = re.sub(r'\n+', '\n', re.sub(r'\n ', '\n', paragraph_file.read()).strip()).splitlines()
                    writer.write({'guid': guid,
                                  'q_paras': query_text,
                                  'c_paras': para_text,
                                  'label': 1 if paragraph.split('.')[0] in doc_rel_id else 0})

