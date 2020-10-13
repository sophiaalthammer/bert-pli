# BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval

This repository contains the code for the reproduction paper **Cross-domain Retrieval in the Legal and Patent Domain: a Reproducability Study**
 of the paper [BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval](https://www.ijcai.org/Proceedings/2020/484) 
 and is based on the [BERT-PLI Github repository](https://github.com/ThuYShao/BERT-PLI-IJCAI2020).

We added the missing data preprocessing scripts as well as the script for fine-tuning the BERT model on binary classification, which
 is based on [HuggingFace' transformers library](https://github.com/huggingface/transformers). Furthermore
 we added scripts for the binary evaluation with the [SciKitLearn classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) as well as for the ranking evaluation 
 using the [pytrec_eval libary](https://github.com/cvangysel/pytrec_eval).


## Outline

### Model

- ``./model/nlp/BertPoolOutMax.py`` : model paragraph-level interactions between documents.

- ``./model/nlp/AttenRNN.py`` : aggregate paragraph-level representations. 

### Config

- ``./config/nlp/BertPoolOutMax.config`` : parameters for ``./model/nlp/BertPoolOutMax.py`` .

- ``./config/nlp/AttenGRU.config`` / ``./config/nlp/AttenLSTM.config`` : parameters for ``./model/nlp/AttenRNN.py`` (GRU / LSTM, repectively)


### Preprocessing

- BERT finetuning


- Pyserini index creation and search


- for training AttenRNN


- poolout to train



### Formatter


- input format for BERT fine-tuning



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

- Eval?



## Data

####Legal datasets

For the legal datasets refer to [COLIEE 2019](https://sites.ualberta.ca/~rabelo/COLIEE2019/), for the paragraph-level 
finetuning of the BERT model the dataset from Task 1 is used, for the document retrieval the dataset from Task 2.


####Patent datasets
For the patent dataset refer to [CLEF-IP 2013](http://www.ifs.tuwien.ac.at/~clef-ip/download-central.shtml), for the paragraph-level
finetuning of the BERT model the dataset from the [claims-to-passage 2013 task](http://www.ifs.tuwien.ac.at/~clef-ip/2013/claims-to-passage.shtml) is used
, for the document retrieval the dataset from the [prior art candidate search 2011 task](http://www.ifs.tuwien.ac.at/~clef-ip/download/2011/index.shtml).
The candidates are retrieved from the [patent corpus](http://www.ifs.tuwien.ac.at/~clef-ip/download-central.shtml) published in 2013.


## Contact
For more details, please refer to our reproduction paper **Cross-domain Retrieval in the Legal and Patent Domain: a Reproducability Study**. If you have any questions, please email sophia.althammer@tuwien.ac.at . 
