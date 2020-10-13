import os
import argparse
import random
import csv
import re
random.seed(42)

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--train-dir', action='store', dest='train_dir',
                    help='training file directory location', required=True)
parser.add_argument('--output-dir', action='store', dest='output_dir',
                    help='output directory location', required=True)

args = parser.parse_args()

#train_dir = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_train'
#output_dir = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1'


#
# load directory structure
#

list_dir = [x for x in os.walk(args.train_dir)]

with open(os.path.join(args.output_dir, 'eval.tsv'), 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')

    for sub_dir in list_dir[0][1]:
        with open(os.path.join(args.train_dir, sub_dir, 'base_case.txt'), 'r') as entailed_fragment:
            query_text = re.sub(r'\n+', '\n', re.sub(r'\n ', '\n', entailed_fragment.read()).strip())

        # read in relevant document ids
        with open(os.path.join(args.train_dir, sub_dir, 'noticed_cases.txt'), 'r') as entailing_paragraphs:
            doc_rel_id = entailing_paragraphs.read().splitlines()

        list_sub_dir_paragraphs = [x for x in os.walk(os.path.join(args.train_dir, sub_dir, 'candidates'))]
        paragraphs_text = {}
        for paragraph in list_sub_dir_paragraphs[0][2]:
            with open(os.path.join(args.train_dir, sub_dir, 'candidates', paragraph), 'r') as paragraph_file:
                para_text = re.sub(r'\n+', '\n',re.sub(r'\n ', '\n', paragraph_file.read()).strip())
                para_id = paragraph.split('.')[0]

                tsv_writer.writerow([sub_dir, para_id, query_text, para_text, 1 if para_id in doc_rel_id else 0])