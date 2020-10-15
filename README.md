# Cross-domain Retrieval in the Legal and Patent Domain: a Reproducability Study

This repository contains the code for the reproduction paper **Cross-domain Retrieval in the Legal and Patent Domain: a Reproducability Study**
 of the paper [BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval](https://www.ijcai.org/Proceedings/2020/484) 
 and is based on the [BERT-PLI Github repository](https://github.com/ThuYShao/BERT-PLI-IJCAI2020).

We added the missing data preprocessing scripts as well as the script for fine-tuning the BERT model on binary classification, which
 is based on [HuggingFace' transformers library](https://github.com/huggingface/transformers). Furthermore
 we added scripts for the binary evaluation with the [SciKitLearn classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) as well as for the ranking evaluation 
 using the [pytrec_eval libary](https://github.com/cvangysel/pytrec_eval).
 
The open-sourced trained models can be found [here](https://zenodo.org/record/4088010).


## Outline

### Model

- ``./model/nlp/BertPoolOutMax.py`` : model paragraph-level interactions between documents.

- ``./model/nlp/AttenRNN.py`` : aggregate paragraph-level representations. 

### Config

- ``./config/nlp/BertPoolOutMax.config`` : parameters for ``./model/nlp/BertPoolOutMax.py`` .

- ``./config/nlp/AttenGRU.config`` / ``./config/nlp/AttenLSTM.config`` : parameters for ``./model/nlp/AttenRNN.py`` (GRU / LSTM, repectively)


### Preprocessing

#### BERT finetuning

required format of the tsv-files:

```
[
    label (0/1),
    claim_id,
    passage_id,
    claim_text,
    passage_text
]
```

- ``./preprocessing/coliee19_task2_create_train_input.py``: preprocessing for the patent train dataset from the COLIEE2019 Task 2 for BERT fine-tuning

```bash
python coliee19_task2_create_train.py  --train-dir /home/data/coliee/task2/train --output-dir /home/data/coliee/task2/ouput 
```

- ``./preprocessing/coliee19_task2_create_test.py``: preprocessing for the patent test dataset from the COLIEE2019 Task 2 for BERT fine-tuning

```bash
python coliee19_task2_create_test.py  --train-dir /home/data/coliee/task2/train --output-dir /home/data/coliee/task2/ouput --test-gold-labels /home/data/coliee/task2/task2_test_golden-labels.xml
```

- ``./preprocessing/coliee19_task2_create_test.py``: preprocessing for the patent train and merged test dataset from the COLIEE2019 Task 2 for BERT fine-tuning

```bash
python coliee19_task2_create_train_test.py --test-dir /home/data/coliee/task2/test --train-dir /home/data/coliee/task2/train --output-dir /home/data/coliee/task2/ouput --test-gold-labels /home/data/coliee/task2/task2_test_golden-labels.xml
```

- ``./preprocessing/clefip13_ctp_create_train.py``: preprocessing for the patent training dataset from the CLEF-IP claim-to-passage task for BERT fine-tuning

```bash
python clefip13_ctp_create_train.py  --train-dir /home/data/clefip/ctp/ --output-dir /home/data/clefip/ctp/ouput --corpus-dir  --corpus-dir  /home/data/clefip/corpus
```

- ``./preprocessing/clefip13_ctp_create_test.py``: preprocessing for the patent test dataset from the CLEF-IP claim-to-passage task for BERT fine-tuning

```bash
python clefip13_ctp_create_test.py  --train-dir /home/data/clefip/ctp/ --output-dir /home/data/clefip/ctp/ouput --corpus-dir  --corpus-dir  /home/data/clefip/corpus
```

#### Filter CLEF-IP prior-art-candidate search datasets for only english topics

- finds the english topics of the train and test files and creates txt-files with the document ids of english topics

```bash
python clefip11_pac_filer_english_topics.py  --train-dir /home/data/clefip/pac/train --train_topics /home/data/clefip/pac/train/files --corpus-dir /home/data/clefip/corpus  --test-dir /home/data/clefip/test/ --test-topics /home/data/clefip/test/files/
```

#### Pyserini index creation and search

[pyserini Github repository](https://github.com/castorini/pyserini) for further explanations.

General setup for both domains:

1. index_jsonl: create JSON-format for the pyserini indexer, either:

    - Folder with files, each of which contains an array of JSON documents 
    - Folder with files, each of which contains a JSON on an individual line (often called JSONL format)

with the following format:
```
{'id': '001', 'contents': 'text'}
```

- ``./preprocessing/coliee19_task1_index_jsonl.py``: json-format for the legal dataset from the COLIEE2019 task 1 for pyserini index creation

```bash
python coliee19_task1_index_jsonl.py  --train-dir /home/data/coliee/task2/corpus/ 
```

- ``./preprocessing/clefip11_pac_index_jsonl.py``: json-format for the patent dataset from the CLEF-IP prior-art-candidate task for pyserini index creation

```bash
python clefip11_pac_index_jsonl.py  --corpus-dir /home/data/clefip/corpus/ --json-dir /home/data/clefip/corpus_json 
```
    

2. create index with pyserini

```
python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -threads 1 -input integrations/resources/sample_collection_jsonl \
 -index indexes/sample_collection_jsonl -storePositions -storeDocvectors -storeRaw
```

3. index_search: search the created index with pyserini for the given topics


- ``./preprocessing/coliee19_task1_index_search.py``: search the legal index for the topics from the COLIEE2019 task 1 for pyserini index search

```bash
python coliee19_task1_index_search.py  --train-dir /home/data/coliee/task2/corpus/ 
```

- ``./preprocessing/clefip11_pac_index_search.py``: search the patent index for the topics from the CLEF-IP prior-art-candidate task for pyserini index search

```bash
python clefip11_pac_index_search.py  --index-dir /home/data/clefip/index/ --topic-dir /home/data/clefip/task1/train/
```


#### Preprocessing for training AttenRNN

create format for the BertDocParaFormatter.py

- ``./preprocessing/coliee19_task1_json_lines.py``: create format for AttentionRNN training dataset from the COLIEE2019 Task 1 

```bash
python coliee19_task1_json_lines.py  --train-dir /home/data/coliee/train/ 
```

- ``./preprocessing/coliee19_task1_json_lines_test.py``: create format for AttentionRNN test dataset from the COLIEE2019 Task 1 

```bash
python coliee19_task1_json_lines_test.py  --train-dir /home/data/coliee/train/ --output-dir /home/data/coliee/output --test-gold-labels /home/data/coliee/task1_test_golden-labels.xml
```


- ``./preprocessing/clefip11_pac_json_lines.py``: create format for AttentionRNN training/test dataset from the COLIEE2019 Task 1 from the CLEF-IP prior-art-candidate task

```bash
python clefip11_pac_json_lines.py  --train-dir /home/data/clefip/train/ --corpus-dir /home/data/clefip/corpus  --folder-name bm25_top50
```



#### Poolout to train

```bash
python poolout_to_train.py  --train-dir /home/data/clefip/train/ 
```


### Formatter

- ``./formatter/nlp/BertDocParaFormatter.py`` : prepare input for ``./model/nlp/BertPoolOutMax.py``
  An example:

```
{
	"guid": "queryID_docID",
	"q_paras": [...], // a list of paragraphs in query case,
	"c_paras": [...], // a list of parameters in candidate case,
	"label": 0, // 0 or 1, denote the relevance
}
```

- ``./formatter/nlp/AttenRNNFormatter.py`` : prepare input for ``./model/nlp/AttenRNN.py``
  An example:

```
{
	"guid": "queryID_docID",
	"res": [[],...,[]], // N * 768, result of BertPoolOutMax,
	 "label": 0, // 0 or 1, denote the relevance
}
```

### Scripts

- ``finetune.py``/``poolout.py``/``train.py``/``test.py``, main entrance for *fine-tuning*,*pooling out*, *training*, and *testing*.

### Requirements

- See ``requirements.txt``

## How to Run?

- Finetune BERT model on paragraph-level interaction binary classification

```bash
python finetune.py --model_name bert-base-uncased --task_name MRPC  --do_train   --do_eval   --data_dir /home/data/   --max_seq_length 512   --per_device_train_batch_size 1   --learning_rate 1e-5   --num_train_epochs 3.0   --save_steps 403   --gradient_accumulation 16   --output_dir /home/data/output
```

- Get paragraph-level interactions by BERT: 

```bash
python3 poolout.py -c config/nlp/BertPoolOutMax.config -g [GPU_LIST] --checkpoint [path of Bert checkpoint] --result [path to save results] 
```

- Train

```bash
python3 train.py -c config/nlp/AttenGRU.config -g [GPU_LIST] 
```

or 

```bash
python3 train.py -c config/nlp/AttenLSTM.config -g [GPU_LIST] 
```

- Test

```bash
python3 test.py -c config/nlp/AttenGRU.config -g [GPU_LIST] --checkpoint [path of Bert checkpoint] --result [path to save results] 
```

or 

```bash
python3 test.py -c config/nlp/AttenLSTM.config -g [GPU_LIST] --checkpoint [path of Bert checkpoint] --result [path to save results] 
```

- Eval

evaluate the recall of the index search for the topics for the COLIEE2019 Task 2

```bash
python coliee19_task1_eval_index.py --train-dir /home/data/coliee/task1/train  --test-gold-labels /home/data/coliee/task1/task1_test_golden-labels.xml
```

evaluate the recall of the index search for the topics for the CLEF-IP 2011 prior-art-candidate search

```bash
python clefip11_pac_eval_index.py --train-dir /home/data/clefip/pac/train  --folder-name bm25_top50
```
 
evaluate the binary classification metrics for the COLIEE2019 of CLEF-IP tasks

```bash
python eval_predictions_binary.py --label-file /home/coliee/task1/test/test.json --pred-file /home/coliee/task1/test/pred.txt
```

evaluate the ranking metrics for COLIEE2019 or CLEF-IP tasks

```bash
python eval_predictions_ranking.py --label-file /home/coliee/task1/test/test.json --pred-file /home/coliee/task1/test/pred.txt
```  



## Data

#### Legal datasets

For the legal datasets refer to [COLIEE 2019](https://sites.ualberta.ca/~rabelo/COLIEE2019/), for the paragraph-level 
finetuning of the BERT model the dataset from Task 1 is used, for the document retrieval the dataset from Task 2.


#### Patent datasets
For the patent dataset refer to [CLEF-IP 2013](http://www.ifs.tuwien.ac.at/~clef-ip/download-central.shtml), for the paragraph-level
finetuning of the BERT model the dataset from the [claims-to-passage 2013 task](http://www.ifs.tuwien.ac.at/~clef-ip/2013/claims-to-passage.shtml) is used
, for the document retrieval the dataset from the [prior art candidate search 2011 task](http://www.ifs.tuwien.ac.at/~clef-ip/download/2011/index.shtml).
The candidates are retrieved from the [patent corpus](http://www.ifs.tuwien.ac.at/~clef-ip/download-central.shtml) published in 2013.


## Contact
For more details, please refer to our reproduction paper **Cross-domain Retrieval in the Legal and Patent Domain: a Reproducability Study**. If you have any questions, please email sophia.althammer@tuwien.ac.at . 
