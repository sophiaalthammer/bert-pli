import os
import argparse
import random
import jsonlines
random.seed(42)

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--train-dir', action='store', dest='train_dir',
                    help='training file directory location', required=True)

args = parser.parse_args()

#train_dir = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_train'


#
# load directory structure
#

list_dir = [x for x in os.walk(args.train_dir)]


for sub_dir in list_dir[0][1]:
    with jsonlines.open(os.path.join(args.train_dir, sub_dir, 'candidates.jsonl'), mode='w') as writer:
        # read in all paragraphs with their names and then choose the relevant ones and sample irrelevant ones!
        list_sub_dir_paragraphs = [x for x in os.walk(os.path.join(args.train_dir, sub_dir, 'candidates'))]
        paragraphs_text = {}
        for paragraph in list_sub_dir_paragraphs[0][2]:
            with open(os.path.join(args.train_dir, sub_dir, 'candidates', paragraph), 'r') as paragraph_file:
                para_text = paragraph_file.read().splitlines()
                writer.write({'id':'{}_{}'.format(sub_dir, paragraph.split('.')[0]),
                           'contents': ' '.join([text.strip().replace('\n', '') for text in para_text]).strip()})