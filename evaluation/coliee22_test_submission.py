import json
import re
import ast
import pytrec_eval
import numpy as np
from evaluation.coliee22_task1_eval import read_predictions, read_predictions_json, format_pred_ranking_to_binary


if __name__ == "__main__":
    pred_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2022/task1/bertpli/output/' \
                'bm25_opt/test/attenlstm/output_test_coliee19.json'

    if 'lstm' in pred_file:
        predictions = read_predictions_json(pred_file, score=False)
    else:
        predictions = read_predictions(pred_file, score=False)

    # pred_cutoff = {}
    # for query_id, ranking in predictions.items():
    #     tuples = [(doc_id, score) for doc_id, score in ranking.items()]
    #     tuples.sort(key=lambda x: x[1], reverse=True)
    #     sorted_dict = {k: v for k, v in tuples}
    #     pred_cutoff.update({query_id: sorted_dict})
    #
    # output_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2022/task1/submission/bertpli_test_runtop50.txt'
    #
    # with open(output_file, 'w') as out_file:
    #     for query_id, value in pred_cutoff.items():
    #         rank = 1
    #         for doc_id, prediction in value.items():
    #             out_file.write('{}'.format(query_id) + '\t' + '{}'.format(doc_id) + '\t' + '{}'.format(
    #                 rank) + '\t' + '{}\n'.format(prediction))
    #             rank += 1


    pred_cutoff = format_pred_ranking_to_binary(predictions, cutoff=5)

    output_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2022/task1/submission/DSSR_03.txt'
    cutoff = 5
    with open(output_file, 'w') as f:
        for query_id, value in pred_cutoff.items():
            i = 0
            for doc_id, label in value.items():
                if i < cutoff:
                    f.write('{}'.format(query_id)+ ' '+ '{}'.format(doc_id) + ' ' + 'DSSR_03\n')
                    i += 1

    input_file2 = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2022/task1/submission/DSSR_01.run'
    output_file2 = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2022/task1/submission/DSSR_01.txt'
    # open run files
    with open(output_file2, 'w') as out_file:
        with open(input_file2, 'r') as f:
            lines = f.readlines()

            for line in lines:
                lined_splitted = line.split(' ')
                query_id = lined_splitted[0]
                doc_id = lined_splitted[1]

                out_file.write('{}'.format(query_id) + ' ' + '{}'.format(doc_id) + ' ' + 'DSSR_01\n')



