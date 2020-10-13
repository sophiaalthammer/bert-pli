# BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval

This repository contains the code for BERT-PLI in our IJCAI-PRICAI 2020 submission: *BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval*

## Outline

### Model

- ``./model/nlp/BertPoolOutMax.py`` : model paragraph-level interactions between documents.

- ``./model/nlp/AttenRNN.py`` : aggregate paragraph-level representations. 

### Config

- ``./config/nlp/BertPoolOutMax.config`` : parameters for ``./model/nlp/BertPoolOutMax.py`` .

- ``./config/nlp/AttenGRU.config`` / ``./config/nlp/AttenLSTM.config`` : parameters for ``./model/nlp/AttenRNN.py`` (GRU / LSTM, repectively)


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

- ``poolout.py``/``train.py``/``test.py``, main entrance for *poolling out*, *training*, and *testing*.

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

## Data

Please refer to [COLIEE 2019](https://sites.ualberta.ca/~rabelo/COLIEE2019/)


## Contact
For more details, please refer to our paper **BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval** (*To Appear*). If you have any questions, please email shaoyq18@mails.tsinghua.edu.cn . 
