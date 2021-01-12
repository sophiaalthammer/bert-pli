import os
import argparse
import random
import pandas as pd
import jsonlines
import re
import ast
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
random.seed(42)


def pred_eval_binary(label_file, pred_file):
    labels = []

    with jsonlines.open(label_file, mode='r') as reader:
        for file in reader:
            labels.append(file)

    label_dict = {}
    for label in labels:
        label_dict.update({label.get('guid'): label.get('label')})

    print(len(labels))
    print(len(label_dict))

    with open(pred_file, 'r') as reader:
        content = reader.read().splitlines()
        predictions = [ast.literal_eval(file) for file in content]

    print(len(predictions))

    pred_dict = {}
    for pred in predictions:
        pred_dict.update({pred.get('guid'): np.argmax(pred.get('res'))})

    files = list(label_dict.keys())
    files.sort()

    label_list = []
    pred_list = []
    for file in files:
        label_list.append(label_dict.get(file))
        if pred_dict.get(file):
            pred_list.append(pred_dict.get(file))
        else:
            pred_list.append(0)

    assert len(label_list) == len(pred_list)

    # manual check if the precision, recall and f1 score of the sklearn binary classification package (for relevant class 1)
    #     return the same evaluatin metrics as when they are computed manually according to the formula on
    #     https://sites.ualberta.ca/~rabelo/COLIEE2020/

    df = pd.DataFrame(list(zip(files, label_list, pred_list)),
                      columns=['file', 'label', 'prediction'])

    df_only_rel = df[df['label'] == 1]
    try:
        no_correctly_retrieved = df_only_rel['prediction'].value_counts()[1]
    except:
        no_correctly_retrieved = 0

    no_relevant = len(df_only_rel)
    try:
        df_only_rel_pred = df[df['prediction'] == 1]
        no_retrieved = len(df_only_rel_pred)
    except:
        no_retrieved = 0

    if no_retrieved != 0:
        precision = no_correctly_retrieved / no_retrieved
    else:
        precision = 0
    recall = no_correctly_retrieved / no_relevant
    if precision + recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    with open(pred_file.split('.txt')[0] + '_eval_bin_200_onlyrel.txt', 'w+') as output:
        output.write(classification_report(label_list, pred_list) + '\n')
        output.write("Precision (for [0,1] class): {} \n".format(precision_score(label_list, pred_list, labels=[0, 1], average = None,
                                                                  pos_label=1)))
        output.write("Recall (for [0,1] class): {}\n".format(recall_score(label_list, pred_list, average= None)))
        output.write("F1 score (for [0,1] class): {} \n".format(f1_score(label_list, pred_list, average=None), end='\n\n'))

        output.write("Manually calculated Precision: {} \n".format(precision))
        output.write("Manually calculated Recall: {}\n".format(recall))
        output.write("Manually calculated F1 score: {} \n".format(f1, end='\n\n'))


def pred_eval_bm25_coliee(label_file, bm25_folder, cutoff):
    labels = []

    with jsonlines.open(label_file, mode='r') as reader:
        for file in reader:
            labels.append(file)

    label_dict = {}
    for label in labels:
        label_dict.update({label.get('guid'): label.get('label')})

    print(len(labels))
    print(len(label_dict))

    pred_dict = {}
    for n in range(1,62):
        with open(os.path.join(bm25_folder, 'bm25_top50_{}.txt'.format('{0:03d}'.format(n))), 'r') as out:
            rank = [text for text in [text.split('\n')[0].strip() for text in out.readlines()]]
            i = 0
            for pred in rank:
                pred_dict.update({pred: 1 if i < cutoff else 0})
                i = i + 1

    files = list(label_dict.keys())
    files.sort()

    label_list = []
    pred_list = []
    for file in files:
        label_list.append(label_dict.get(file))
        if pred_dict.get(file):
            pred_list.append(pred_dict.get(file))
        else:
            pred_list.append(0)

    assert len(label_list) == len(pred_list)

    #print(classification_report(label_list, pred_list))

    # These values are the same as in the table above
    #print("Precision (micro): %f" % precision_score(label_list, pred_list, labels=[0, 1], average='micro', pos_label=1))
    #print("Recall (micro):    %f" % recall_score(label_list, pred_list, average='micro'))
    #print("F1 score (micro):  %f" % f1_score(label_list, pred_list, average='micro'), end='\n\n')

    #print("Precision (macro): %f" % precision_score(label_list, pred_list, labels=[0, 1], average='macro', pos_label=1))
    #print("Recall (macro):    %f" % recall_score(label_list, pred_list, average='macro'))
    #print("F1 score (macro):  %f" % f1_score(label_list, pred_list, average='macro'), end='\n\n')

    with open(os.path.join(bm25_folder, 'eval_bin_200_onlyrel.txt'), 'w+') as output:
        output.write(classification_report(label_list, pred_list) + '\n')
        output.write(
            "Precision (for [0,1] class): {} \n".format(precision_score(label_list, pred_list, labels=[0, 1], average=None,
                                                              pos_label=1)))
        output.write("Recall (for [0,1] class): {}\n".format(recall_score(label_list, pred_list, average=None)))
        output.write("F1 score (for [0,1] class): {} \n".format(f1_score(label_list, pred_list, average=None), end='\n\n'))
        #output.write("Precision (macro): {} \n".format(
        #    precision_score(label_list, pred_list, labels=[0, 1], average='macro', pos_label=1)))
        #output.write("Recall (macro):    {} \n".format(recall_score(label_list, pred_list, average='macro')))
        #output.write("F1 score (macro):  {} \n".format(f1_score(label_list, pred_list, average='macro'), end='\n\n'))

    return precision_score(label_list, pred_list, labels=[0, 1], average='macro', pos_label=1), recall_score(label_list, pred_list, average='macro'), f1_score(label_list, pred_list, average='macro')



def pred_eval_bm25_clef(label_file, bm25_folder, cutoff):
    labels = []

    with jsonlines.open(label_file, mode='r') as reader:
        for file in reader:
            labels.append(file)

    label_dict = {}
    for label in labels:
        label_dict.update({label.get('guid'): label.get('label')})

    print(len(labels))
    print(len(label_dict))

    keys = list(set([key.split('_')[0] for key in list(label_dict.keys())]))

    pred_dict = {}
    for key in keys:
        with open(os.path.join(bm25_folder, 'bm25_top50_{}.xml.txt'.format(key)), 'r') as out:
            rank = [text.split('-')[0] + '-' + text.split('-')[1] for text in
                    [text.split('\n')[0].strip() for text in out.readlines()]]
            i = 0
            for pred in rank:
                pred_dict.update({'{}_{}'.format(key, pred): 1 if i < cutoff else 0})
                i = i + 1

    files = list(label_dict.keys())
    files.sort()

    label_list = []
    pred_list = []
    for file in files:
        label_list.append(label_dict.get(file))
        if pred_dict.get(file):
            pred_list.append(pred_dict.get(file))
        else:
            pred_list.append(0)

    assert len(label_list) == len(pred_list)

    with open(os.path.join(bm25_folder, 'eval_bin_200_onlyrel.txt'), 'w+') as output:
        output.write(classification_report(label_list, pred_list) + '\n')
        output.write(
            "Precision (for [0,1] class): {} \n".format(precision_score(label_list, pred_list, labels=[0, 1], average=None,
                                                              pos_label=1)))
        output.write("Recall (for [0,1] class): {}\n".format(recall_score(label_list, pred_list, average=None)))
        output.write("F1 score (for [0,1] class): {} \n".format(f1_score(label_list, pred_list, average=None), end='\n\n'))
        #output.write("Precision (macro): {} \n".format(
        #    precision_score(label_list, pred_list, labels=[0, 1], average='macro', pos_label=1)))
        #output.write("Recall (macro):    {} \n".format(recall_score(label_list, pred_list, average='macro')))
        #output.write("F1 score (macro):  {} \n".format(f1_score(label_list, pred_list, average='macro'), end='\n\n'))

    return precision_score(label_list, pred_list, labels=[0, 1], average='macro', pos_label=1), recall_score(label_list, pred_list, average='macro'), f1_score(label_list, pred_list, average='macro')



if __name__ == "__main__":
    #
    # config
    #
    #parser = argparse.ArgumentParser()

    #parser.add_argument('--label-file', action='store', dest='label_file',
    #                    help='json file with the labels of the test file', required=True)
    #parser.add_argument('--pred-file', action='store', dest='pred_file',
    #                    help='txt file with the binary predictions of the test file', required=False)
    #parser.add_argument('--bm25-folder', action='store', dest='bm25_folder',
    #                    help='folder with the BM25 retrieval per guid which the result is compared to', required=False)
    #parser.add_argument('--cutoff', action='store', dest='cutoff',
    #                    help='cutoff value for BM25 prediction', required=False)

    #args = parser.parse_args()

    label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/test_org_200.json'
    pred_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_lawbert_lawattenlstm.txt'

    predictions = ['/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_lawbert_lawattenlstm.txt',
                        '/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_lawbert_lawattengru.txt',
                        '/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_lawbert_patentattenlstm.txt',
                        '/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_lawbert_patentattengru.txt',
                        '/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_patentbert_lawattenlstm2.txt',
                        '/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_patentbert_lawattengru2.txt',
                        '/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_patentbert_patentattenlstm.txt',
                        '/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_patentbert_patentattengru.txt',
                        '/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_top50_wogold_bertorg_lawattenlstm.txt',
                        '/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_top50_wogold_bertorg_lawattengru.txt',
                        '/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_top50_wogold_bertorg_patentattenlstm.txt',
                        '/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_top50_wogold_bertorg_patentattengru.txt']

    bm25_folder = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2019/task1/task1_test/task1_test_bm25_top50'
    cutoff = 5

    #label_file = '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/test_org_top50_wogold.json'

    #predictions = ['/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output/output_test_top50_wogold_lawbert_lawattenlstm.txt',
    #               '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output/output_test_top50_wogold_lawbert_lawattengru.txt',
    #               '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output/output_test_top50_wogold_lawbert_patentattenlstm.txt',
    #               '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output/output_test_top50_wogold_lawbert_patentattengru.txt',
    #               '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output/output_test_top50_wogold_patentbert_lawattenlstm.txt',
    #               '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output/output_test_top50_wogold_patentbert_lawattengru.txt',
    #               '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output/output_test_top50_wogold_patentbert_patentattenlstm.txt',
    #               '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output/output_test_top50_wogold_patentbert_patentattengru.txt',
    #               '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output/output_test_top50_wogold_bertorg_lawattenlstm.txt',
    #               '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output/output_test_top50_wogold_bertorg_lawattengru.txt',
    #               '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output/output_test_top50_wogold_bertorg_patentattenlstm.txt',
    #               '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output/output_test_top50_wogold_bertorg_patentattengru.txt']

    #bm25_folder = '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/bm25_top50'
    #cutoff = 5

    for pred_file in predictions:
        pred_eval_binary(label_file, pred_file)

    #if bm25_folder and cutoff:
        #prec, recall, f1 = pred_eval_bm25_coliee(label_file, bm25_folder, cutoff)
    #    pred_eval_bm25_clef(label_file, bm25_folder, cutoff)

