import os
import csv
import argparse
import random
random.seed(42)
#
# creates balanced training input file for finetuning bert with 1 or 0 labels
#


#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--train-dir', action='store', dest='train_dir',
                    help='training file directory location', required=True)
parser.add_argument('--output-dir', action='store', dest='output_dir',
                    help='training output file location for train.tsv', required=True)

args = parser.parse_args()

#train_dir = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task2/task2_train'
#output_dir = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task2'

#
# load directory structure
#

list_dir = [x for x in os.walk(args.train_dir)]

#
# Write train.tsv file with query_text \t doc_relevant_text \t doc_irrelevant_text for all subdirectories
#

with open(os.path.join(args.output_dir, 'train.tsv'), 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')

    samples = []

    for sub_dir in list_dir[0][1]:
        # read in query text
        with open(os.path.join(args.train_dir, sub_dir, 'entailed_fragment.txt'), 'r') as entailed_fragment:
            query_text_lines = entailed_fragment.read().splitlines()
            query_text = ' '.join([text.strip().replace('\n', '') for text in query_text_lines])
        # read in relevant document ids
        with open(os.path.join(args.train_dir, sub_dir, 'entailing_paragraphs.txt'), 'r') as entailing_paragraphs:
            doc_rel_id = entailing_paragraphs.read().splitlines()

        # read in all paragraphs with their names and then choose the relevant ones and sample irrelevant ones!
        list_sub_dir_paragraphs = [x for x in os.walk(os.path.join(args.train_dir, sub_dir, 'paragraphs'))]
        paragraphs_text = {}
        for paragraph in list_sub_dir_paragraphs[0][2]:
            with open(os.path.join(args.train_dir, sub_dir, 'paragraphs', paragraph), 'r') as paragraph_file:
                para_text = paragraph_file.read().splitlines()[1:]
                paragraphs_text.update({paragraph.split('.')[0]: ' '.join([text.strip().replace('\n', '') for text in para_text])})

        # sample randomly from the paragraphs as many irrelevant paragraphs as you have relevant ones
        doc_irrel_id = random.sample([irrel_id for irrel_id in paragraphs_text.keys() if irrel_id not in doc_rel_id], len(doc_rel_id))

        assert len(doc_irrel_id) == len(doc_rel_id)

        # create mapping of training pairs with one relevant and one irrelevant text
        ids_rel_irrel = list(zip(doc_rel_id, doc_irrel_id))

        # write sample
        for i in range(len(ids_rel_irrel)):
            doc_rel_text = paragraphs_text.get(ids_rel_irrel[i][0])
            doc_irrel_text = paragraphs_text.get(ids_rel_irrel[i][1])
            samples.append([1, sub_dir, ids_rel_irrel[i][0], query_text, doc_rel_text])
            samples.append([0, sub_dir, ids_rel_irrel[i][1], query_text, doc_irrel_text])

        # important: shuffle the train data
        random.shuffle(samples)

        # write text in train.tsv file
        for sample in samples:
            tsv_writer.writerow(sample)


