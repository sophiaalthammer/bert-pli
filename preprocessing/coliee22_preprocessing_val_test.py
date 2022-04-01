import os
import random
import json
import jsonlines
import re
random.seed(42)

def read_label_file(label_file: str):
    with open(label_file, 'rb') as f:
        labels = json.load(f)

    # other format of labels:
    qrels = {}
    for key, values in labels.items():
        val_format = {}
        for value in values:
            val_format.update({'{}'.format(value.split('.')[0]): 1})
        qrels.update({key.split('.')[0]: val_format})
    return qrels


def read_in_run_bmopt(run_path):
    with open(run_path, 'r') as f:
        lines = f.readlines()

        run = {}
        for line in lines:
            lined_splitted = line.split(' 0 ')
            query_id = lined_splitted[0]
            doc_id = lined_splitted[1]
            if run.get(query_id):
                run.get(query_id).update({rank: doc_id})
                rank += 1
            else:
                run.update({query_id:{}})
                rank = 1
                run.get(query_id).update({rank: doc_id})
                rank += 1
    return run

def read_in_paragraph_file(file_path):
    with open(file_path, 'r') as json_file:
        json_list = list(json_file)

    data = {}
    for json_str in json_list:
        result = json.loads(json_str)
        doc_id = result.get('id').split('_')[0]
        text = result.get('contents')
        if data.get(doc_id):
            p_list = data.get(doc_id)
            p_list.append(text)
            data.update({doc_id: p_list})
        else:
            data.update({doc_id:[text]})
    return data

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def write_json_bertpli_file(run_val, corpus_docs, val_queries, top_n, output_dir, qrels_val=None, parts=6):
    list_query_ids = list(split(list(run_val.keys()), parts))

    for j in range(parts):
        name = 'run_for_bertpli_top{}_{}_{}.json'.format(top_n[0], top_n[1], j)
        query_ids_list = list_query_ids[j]
        with jsonlines.open(os.path.join(output_dir, name), mode='w') as writer:
            for query_id in query_ids_list:
                value = run_val.get(query_id)
                for rank, doc_id in value.items():
                    if top_n[0] <= rank <= top_n[1]:
                        guid = '{}_{}'.format(query_id, doc_id)
                        if qrels_val:
                            writer.write({'guid': guid,
                                          'q_paras': val_queries.get(query_id),
                                          'c_paras': corpus_docs.get(doc_id),
                                          'label': 1 if doc_id in list(qrels_val.get(query_id).keys()) else 0})
                        else:
                            writer.write({'guid': guid,
                                          'q_paras': val_queries.get(query_id),
                                          'c_paras': corpus_docs.get(doc_id),
                                          'label': 0})

def read_run_bm25(run_file):
    with open(run_file, 'r') as f:
        lines = f.readlines()

        run = {}
        for line in lines:
            lined_splitted = line.split(' ')
            query_id = lined_splitted[0].split('_0')[0]
            doc_id = lined_splitted[2].split('_0')[0]
            rank = int(lined_splitted[3])
            if run.get(query_id):
                run.get(query_id).update({rank: doc_id})
            else:
                run.update({query_id:{}})
                run.get(query_id).update({rank: doc_id})
    return run


def read_in_run_bm25opt2(run_path):
    with open(run_path, 'r') as f:
        lines = f.readlines()

        run = {}
        for line in lines:
            lined_splitted = line.split('\t')
            query_id = lined_splitted[0]
            if query_id == 'qid':
                pass
            else:
                doc_id = lined_splitted[2]
                rank = int(lined_splitted[3])
                if run.get(query_id):
                    run.get(query_id).update({rank: doc_id})
                else:
                    run.update({query_id: {}})
                    run.get(query_id).update({rank: doc_id})
    return run


if __name__ == "__main__":
    # read in the run
    #run_path_val = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2022/task1/val/coliee2021_test_bm25_optimized_top30_with_new_terms_f12005.run'
    #run_path_test = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2022/task1/test/coliee22_bm25_optimized_with_new_terms.run'
    run_path_val = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2022/task1/bm25_opt/val/top_of_optimizedbm100.run'
    run_path_test = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2022/task1/bm25_opt/test/test2022_top100.csv'

    run_val = read_in_run_bm25opt2(run_path_val)
    run_test = read_in_run_bm25opt2(run_path_test)

    # read in the cases with their paragraph splits
    # read in query cases
    queries_test_path = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2022/task1/test/corpus_separately_para_with_intro_summ.jsonl'
    queries_val_path = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/separately_para_w_summ_intro.jsonl'

    qrels_val = read_label_file('/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/task1_test_labels_2021.json')

    # read in id content file
    val_queries = read_in_paragraph_file(queries_val_path)
    test_queries = read_in_paragraph_file(queries_test_path)

    # read in corpus
    corpus_path = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/corpus_jsonl/separately_para_w_summ_intro.jsonl'
    corpus_docs = read_in_paragraph_file(corpus_path)

    # first create the files for top10 for reranking
    # then for top10-15 for reranking!
    output_dir = '/'.join(run_path_val.split('/')[:-1])
    top_n = [1,50]
    write_json_bertpli_file(run_val, corpus_docs, val_queries, top_n, output_dir, qrels_val, parts=4)

    #top_n = [11,15]
    #write_json_bertpli_file(run_val, corpus_docs, val_queries, top_n, output_dir, qrels_val, parts)

    # test
    output_dir = '/'.join(run_path_test.split('/')[:-1])
    top_n = [1, 50]
    write_json_bertpli_file(run_test, test_queries, test_queries, top_n, output_dir, parts=4)

    #top_n = [11, 15]
    #write_json_bertpli_file(run_test, corpus_docs, test_queries, top_n, output_dir, parts=6)

    # include separation of files, so that new ones are created!

    # create training data from coliee2021 train
    queries_train_path = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/train/separately_para_w_summ_intro.jsonl'
    queries_val_path = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/separately_para_w_summ_intro.jsonl'
    qrels_train = read_label_file('/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/train/train_labels.json')

    # read in id content file
    train_queries = read_in_paragraph_file(queries_train_path)
    val_queries = read_in_paragraph_file(queries_val_path)

    # all train queries from 2021!
    train_queries.update(val_queries)

    # read in corpus
    corpus_path = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/corpus_jsonl/separately_para_w_summ_intro.jsonl'
    corpus_docs = read_in_paragraph_file(corpus_path)

    # run from bm25, then use top n for training!
    bm25_run_train = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/search/train/search_whole_doc_w_summ_intro_whole_doc_w_summ_intro.txt'
    bm25_run_val = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/search/val/search_whole_doc_w_summ_intro_whole_doc_w_summ_intro.txt'

    run_train = read_run_bm25(bm25_run_train)
    run_val = read_run_bm25(bm25_run_val)
    run_train.update(run_val)

    output_dir = '/'.join(queries_train_path.split('/')[:-1])
    top_n = [1,15]
    parts = 6
    write_json_bertpli_file(run_train, corpus_docs, train_queries, top_n, output_dir, qrels_train, parts)

    # i also need the positives which are not in bm25! i think so?
    # add the ones from qrels_train which are not in train set

    assert run_train.keys() == qrels_train.keys()

    rel_docs_per_query = {}
    for key, value in qrels_train.items():
        rel_docs = list(value.keys())
        rel_docs_per_query.update({key:[]})
        for doc in rel_docs:
            if doc not in list(run_train.get(key).values()):
                list1 = rel_docs_per_query.get(key)
                list1.append(doc)
                rel_docs_per_query.update({key:list1})

    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2022/task1/train/train_pos_leftovers.json'
    with jsonlines.open(os.path.join(output_dir), mode='w') as writer:
        for query_id, value in rel_docs_per_query.items():
            for doc_id in value:
                guid = '{}_{}'.format(query_id, doc_id)
                writer.write({'guid': guid,
                              'q_paras': train_queries.get(query_id),
                              'c_paras': corpus_docs.get(doc_id),
                              'label': 1})





