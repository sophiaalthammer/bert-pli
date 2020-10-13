import os
import argparse
import random
from pyserini.search import SimpleSearcher
import jsonlines
random.seed(42)

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--train-dir', action='store', dest='train_dir',
                    help='train directory location', required=True)


args = parser.parse_args()

#train_dir = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_train'

#
# load directory structure
#

list_dir = [x for x in os.walk(args.train_dir)]

with open(os.path.join(args.train_dir, 'failed_dirs.txt'),'w') as failed_dir:
    for sub_dir in list_dir[0][1]:
        try:
            # read in query text
            with open(os.path.join(args.train_dir, sub_dir, 'base_case.txt'), 'r') as entailed_fragment:
                query_text_lines = entailed_fragment.read().splitlines()
                query_text = ' '.join([text.strip().replace('\n', '') for text in query_text_lines][:250]).strip()

            searcher = SimpleSearcher(os.path.join(args.train_dir, sub_dir, 'index'))
            searcher.set_bm25(0.9, 0.4)

            hits = searcher.search(query_text, 50)

            # Print the first 50 hits:
            with open(os.path.join(args.train_dir, sub_dir, 'bm25_top50.txt'),"w",encoding="utf8") as out_file:
                for i in range(0, 50):
                    out_file.write(f'{hits[i].docid:55}\n')
                    print(f'{hits[i].docid:55}')
        except:
            print('failed for {}'.format(sub_dir))
            failed_dir.write('{}\n'.format(sub_dir))