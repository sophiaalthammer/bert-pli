import os
import argparse
import random
import jsonlines
import re
random.seed(42)

#
# config
#
#parser = argparse.ArgumentParser()

#parser.add_argument('--train-dir', action='store', dest='train_dir',
#                    help='training file directory location', required=True)


#args = parser.parse_args()

train_dir = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_train'


#
# load directory structure
#

list_dir = [x for x in os.walk(train_dir)]

with jsonlines.open(os.path.join(train_dir, 'train_org.json'), mode='w') as writer:
    #for sub_dir in list_dir[0][1]:
    sub_dir = '001'
    with open(os.path.join(train_dir, sub_dir, 'base_case.txt'), 'r') as entailed_fragment:
        query_text = re.sub(r'\n+', '\n', re.sub(r'\n ', '\n', entailed_fragment.read()).strip()).splitlines()


    # read in relevant document ids
    with open(os.path.join(train_dir, sub_dir, 'noticed_cases.txt'), 'r') as entailing_paragraphs:
        doc_rel_id = entailing_paragraphs.read().splitlines()

    # read in bm25 top 50
    with open(os.path.join(train_dir, sub_dir, 'bm25_top50.txt'), 'r') as entailing_para:
        bm25_top50 = entailing_para.read().splitlines()
        bm25_top50 = [doc.strip() for doc in bm25_top50]

    # read in all paragraphs with their names and then choose the relevant ones and sample irrelevant ones!
    list_sub_dir_paragraphs = [x for x in os.walk(os.path.join(
        train_dir, sub_dir, 'candidates'))]
    paragraphs_text = {}
    for paragraph in list_sub_dir_paragraphs[0][2]:
        guid = '{}_{}'.format(sub_dir, paragraph.split('.')[0])
        if guid in bm25_top50:
            with open(os.path.join(train_dir, sub_dir, 'candidates', paragraph), 'r') as paragraph_file:
                para_text = re.sub(r'\n+', '\n', re.sub(r'\n ', '\n', paragraph_file.read()).strip()).splitlines()
                writer.write({'guid': guid,
                              'q_paras': query_text,
                              'c_paras': para_text,
                              'label': 1 if paragraph.split('.')[0] in doc_rel_id else 0})
