import os
import argparse
import random
import jsonlines
import re
import ast
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
random.seed(42)
import argparse
import os
import sys
import pytrec_eval
from scipy import stats


def ranking_metrics_per_topic(label_file, pred_file):
    #
    # load directory structure
    #

    labels = []

    with jsonlines.open(label_file, mode='r') as reader:
        for file in reader:
            labels.append(file)
    label_dict = {}
    for label in labels:
        label_dict.update({label.get('guid'): label.get('label')})


    with open(pred_file, 'r') as reader:
        content = reader.read().splitlines()
        predictions = [ast.literal_eval(file) for file in content]
    pred_dict = {}
    pos = []
    for pred in predictions:
        pred_dict.update({pred.get('guid'): pred.get('res')[1]}) # für binary ist hier 1 anstatt 0 (für mseloss output)
        pos.append(pred.get('res')[1]) # für binary ist hier 1 anstatt 0 (für mseloss)


    files = list(pred_dict.keys())
    files.sort()

    qrels = {}
    for file in files:
        qrels.update({file.split('_')[0]: {}})
    for file in files:
        #print(file.split('_')[0])
        qrels.get(file.split('_')[0]).update({file.split('_')[1]: label_dict.get(file)})

    run = {}
    for file in files:
        run.update({file.split('_')[0]: {}})
    for file in files:
        #print(file.split('_')[0])
        run.get(file.split('_')[0]).update({file.split('_')[1]: pred_dict.get(file) + min(pos)})

    # trec eval

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, pytrec_eval.supported_measures)

    results = evaluator.evaluate(run)

    list_topic = []
    map = []
    ndcg5 = []
    ndcg10 = []
    ndcg20 = []
    for query_id, query_measures in sorted(results.items()):
        list_topic.append(query_id)
        for measure, value in sorted(query_measures.items()):
            if measure == 'map':
                map.append(value)
            if measure == 'ndcg_cut_5':
                ndcg5.append(value)
            if measure == 'ndcg_cut_10':
                ndcg10.append(value)
            if measure == 'ndcg_cut_20':
                ndcg20.append(value)

    assert len(map) == len(list_topic) == len(ndcg5) == len(ndcg10) == len(ndcg20)

    return list_topic, map, ndcg5, ndcg10, ndcg20


def ranking_metrics_per_topic_bm25(label_file, pred_file, bm25_folder):
    #
    # load directory structure
    #

    labels = []

    with jsonlines.open(label_file, mode='r') as reader:
        for file in reader:
            labels.append(file)

    label_dict = {}
    for label in labels:
        label_dict.update({label.get('guid'): label.get('label')})

    with open(pred_file, 'r') as reader:
        content = reader.read().splitlines()
        predictions = [ast.literal_eval(file) for file in content]

    pred_dict = {}
    pos = []
    for pred in predictions:
        pred_dict.update({pred.get('guid'): pred.get('res')[1]})
        pos.append(pred.get('res')[1])

    files = list(pred_dict.keys())
    files.sort()

    qrels = {}
    for file in files:
        qrels.update({file.split('_')[0]: {}})
    for file in files:
        #print(file.split('_')[0])
        qrels.get(file.split('_')[0]).update({file.split('_')[1]: label_dict.get(file)})

    run = {}
    for file in files:
        run.update({file.split('_')[0]: {}})
    for key in list(run.keys()):
        with open(os.path.join(bm25_folder, 'bm25_top50_{}.txt'.format(key)), 'r') as out:  #.xml weg für coliee corpus!
            text = [text.split('_')[1] for text in
                    [text.split('\n')[0].strip() for text in out.readlines()]]
            #text = [text.split('-')[0] + text.split('-')[1] for text in
            #        [text.split('\n')[0].strip() for text in out.readlines()]]
            i = 0
            for doc in text:
                run.get(key).update({doc: len(text) - i})
                i = i + 1

    # trec eval
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels, pytrec_eval.supported_measures)

    results = evaluator.evaluate(run)

    list_topic_bm25 = []
    map_bm25 = []
    ndcg5_bm25 = []
    ndcg10_bm25 = []
    ndcg20_bm25 = []
    for query_id, query_measures in sorted(results.items()):
        list_topic_bm25.append(query_id)
        for measure, value in sorted(query_measures.items()):
            if measure == 'map':
                map_bm25.append(value)
            if measure == 'ndcg_cut_5':
                ndcg5_bm25.append(value)
            if measure == 'ndcg_cut_10':
                ndcg10_bm25.append(value)
            if measure == 'ndcg_cut_20':
                ndcg20_bm25.append(value)

    assert len(map_bm25) == len(list_topic_bm25) == len(ndcg5_bm25) == len(ndcg10_bm25) == len(ndcg20_bm25)

    return list_topic_bm25, map_bm25, ndcg5_bm25, ndcg10_bm25, ndcg20_bm25


def paired_ttest(data1, data2):
    stat, p = stats.ttest_ind(data1, data2)
    #print('stat=%.3f, p=%.3f' % (stat, p))
    return p


def ttest_pred_bm25(label_file, pred_file, bm25_folder):
    list_topics, map, ndcg5, ndcg10, ndcg20 = ranking_metrics_per_topic(label_file, pred_file)
    list_topics_bm25, map_bm25, ndcg5_bm25, ndcg10_bm25, ndcg20_bm25 = ranking_metrics_per_topic_bm25(label_file,
                                                                                                      pred_file,
                                                                                                      bm25_folder)
    assert list_topics == list_topics_bm25

    p_map = paired_ttest(map, map_bm25)
    p_ndcg5 = paired_ttest(ndcg5, ndcg5_bm25)
    p_ndcg10 = paired_ttest(ndcg10, ndcg10_bm25)
    p_ndcg20 = paired_ttest(ndcg20, ndcg20_bm25)

    text = '{:25s}{:8f}\n'.format('map', p_map) + '{:25s}{:8f}\n'.format('ndcg_cut_5', p_ndcg5) +\
           '{:25s}{:8f}\n'.format('ndcg_cut_10', p_ndcg10) +'{:25s}{:8f}'.format('ndcg_cut_20', p_ndcg20)

    print(text)
    with open(pred_file.split('.txt')[0] + '_ttest_bm25.txt', 'w') as output:
        output.write(text)


def ttest_pred1_pred2(label_file, pred_file1,  pred_file2):
    list_topics, map, ndcg5, ndcg10, ndcg20 = ranking_metrics_per_topic(label_file, pred_file1)
    list_topics_bm25, map_bm25, ndcg5_bm25, ndcg10_bm25, ndcg20_bm25 = ranking_metrics_per_topic(label_file, pred_file2)

    assert list_topics == list_topics_bm25

    p_map = paired_ttest(map, map_bm25)
    p_ndcg5 = paired_ttest(ndcg5, ndcg5_bm25)
    p_ndcg10 = paired_ttest(ndcg10, ndcg10_bm25)
    p_ndcg20 = paired_ttest(ndcg20, ndcg20_bm25)

    text = '{:25s}{:8f}\n'.format('map', p_map) + '{:25s}{:8f}\n'.format('ndcg_cut_5', p_ndcg5) +\
           '{:25s}{:8f}\n'.format('ndcg_cut_10', p_ndcg10) +'{:25s}{:8f}'.format('ndcg_cut_20', p_ndcg20)

    print(text)
    with open(pred_file1.split('.txt')[0] + pred_file2.split('.txt')[0].split('/')[12] + '_ttest.txt', 'w') as output:
        output.write(text)


if __name__ == "__main__":
    #
    # config
    #
    parser = argparse.ArgumentParser()

    parser.add_argument('--label-file', action='store', dest='label_file',
                        help='org file with the guid and the labels', required=True)
    parser.add_argument('--pred-file', action='store', dest='pred_file',
                        help='file with the binary prediction per guid', required=True)
    parser.add_argument('--pred-file2', action='store', dest='pred_file2',
                        help='file with the binary prediction per guid to be compared with', required=False)
    parser.add_argument('--bm25-folder', action='store', dest='bm25_folder',
                        help='folder with the BM25 retrieval per guid which the result is compared to', required=False)

    args = parser.parse_args()

    #label_file = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/test_org_top50_wogold.json'
    #pred_file = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output/output_test_lawbert_patentattengru.txt'
    #pred_file2 = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/output/output_test_lawbert_lawattengru.txt'

    #label_file = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test/test_org.json'
    #pred_file = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_top50_wogold_bertorg_lawattengru.txt'
    #pred_file2 = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test/output/output_colieedata_test_top50_wogold_bertorg_patentattengru.txt'

    # bm25 labels: from bm25_folder
    #bm25_folder = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/bm25_top20_250'
    #bm25_folder = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test/task1_test_bm25_top50'

    if args.bm25_folder:
        ttest_pred_bm25(args.label_file, args.pred_file, args.bm25_folder)
    if args.pred_file2:
        ttest_pred1_pred2(args.label_file, args.pred_file, args.pred_file2)

