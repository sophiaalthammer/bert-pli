#!/bin/bash
#conda activate bert-pli-env
python coliee22_poolout_to_train.py --run_before_poolout  /newstorage5/salthamm/coliee22/task1/train/run_for_bertpli_top1_15_0.json --run_after_poolout  /newstorage5/salthamm/coliee22/task1/bertpli/poolout_output/train/poolout_bertorg_top1_15_0.jsonwithlabels.json
wait
python coliee22_poolout_to_train.py --run_before_poolout  /newstorage5/salthamm/coliee22/task1/train/run_for_bertpli_top1_15_1.json --run_after_poolout  /newstorage5/salthamm/coliee22/task1/bertpli/poolout_output/train/poolout_bertorg_top1_15_1.jsonwithlabels.json
wait
python coliee22_poolout_to_train.py --run_before_poolout  /newstorage5/salthamm/coliee22/task1/train/run_for_bertpli_top1_15_2.json --run_after_poolout  /newstorage5/salthamm/coliee22/task1/bertpli/poolout_output/train/poolout_bertorg_top1_15_2.jsonwithlabels.json
wait
python coliee22_poolout_to_train.py --run_before_poolout  /newstorage5/salthamm/coliee22/task1/train/run_for_bertpli_top1_15_3.json --run_after_poolout  /newstorage5/salthamm/coliee22/task1/bertpli/poolout_output/train/poolout_bertorg_top1_15_3.jsonwithlabels.json
wait
