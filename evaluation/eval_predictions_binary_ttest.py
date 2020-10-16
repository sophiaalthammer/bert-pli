import os
import argparse
import random
import jsonlines
import re
import ast
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import stats
random.seed(42)


def paired_ttest_bm25(pred_file, bm25_folder, cutoff=6):
    with open(pred_file, 'r') as reader:
        content = reader.read().splitlines()
        predictions = [ast.literal_eval(file) for file in content]

    print(len(predictions))

    label_dict = {}
    for pred in predictions:
        label_dict.update({pred.get('guid'): np.argmax(pred.get('res'))})

    pred_dict = {}
    for n in range(1, 62):
        with open(os.path.join(bm25_folder, 'bm25_top50_{}.txt'.format('{0:03d}'.format(n))), 'r') as out:
            rank = [text for text in [text.split('\n')[0].strip() for text in out.readlines()]]
            i = 0
            for pred in rank:
                pred_dict.update({pred: 1 if i < cutoff else 0})
                i = i + 1

    files = list(pred_dict.keys())
    files.sort()

    label_list = []
    pred_list = []
    for file in files:
        label_list.append(label_dict.get(file))
        pred_list.append(pred_dict.get(file))

    assert len(label_list) == len(pred_list)

    stat, p = stats.ttest_rel(label_list, pred_list)

    print('pred file {} with BM25 binary comparison: pvalue is {}'.format(pred_file.split('/')[12], p))



if __name__ == "__main__":
    #
    # config
    #
    parser = argparse.ArgumentParser()

    parser.add_argument('--pred-file', action='store', dest='pred_file',
                        help='txt file with the binary predictions of the test file', required=True)
    parser.add_argument('--bm25-folder', action='store', dest='bm25_folder',
                        help='folder with the BM25 retrieval per guid which the result is compared to', required=True)
    parser.add_argument('--cutoff', action='store', dest='cutoff',
                        help='cutoff value for BM25 prediction', required=True)

    args = parser.parse_args()

    #label_file = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test/test_org.json'
    #pred_file = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_top50_wogold_bertorg_lawattenlstm.txt'

    #label_file = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/val_org_top20.json'
    #pred_file = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output_test_patentbert_lawattenlstm2.txt'

    #bm25_folder = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test/task1_test_bm25_top50'
    #cutoff = 6

    paired_ttest_bm25(args.pred_file, args.bm25_folder, args.cutoff)


