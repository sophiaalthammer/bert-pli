#!/bin/bash
#conda activate bert-pli-env
python coliee22_poolout_to_train.py --run_before_poolout  /newstorage5/salthamm/coliee22/task1/bm25_opt/test/run_for_bertpli_top1_50_0.json --run_after_poolout  /newstorage5/salthamm/coliee22/task1/bertpli/poolout_output/bm25_opt/test/poolout_bertorg_top1_50_0.json
wait
python coliee22_poolout_to_train.py --run_before_poolout  /newstorage5/salthamm/coliee22/task1/bm25_opt/test/run_for_bertpli_top1_50_1.json --run_after_poolout  /newstorage5/salthamm/coliee22/task1/bertpli/poolout_output/bm25_opt/test/poolout_bertorg_top1_50_1.json
wait
python coliee22_poolout_to_train.py --run_before_poolout  /newstorage5/salthamm/coliee22/task1/bm25_opt/test/run_for_bertpli_top1_50_2.json --run_after_poolout  /newstorage5/salthamm/coliee22/task1/bertpli/poolout_output/bm25_opt/test/poolout_bertorg_top1_50_2.json
wait
python coliee22_poolout_to_train.py --run_before_poolout  /newstorage5/salthamm/coliee22/task1/bm25_opt/test/run_for_bertpli_top1_50_3.json --run_after_poolout  /newstorage5/salthamm/coliee22/task1/bertpli/poolout_output/bm25_opt/test/poolout_bertorg_top1_50_3.json
wait
