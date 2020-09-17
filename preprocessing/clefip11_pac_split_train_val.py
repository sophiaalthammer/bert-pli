import os
import argparse
import random
import jsonlines
from xml.etree import ElementTree as ET
random.seed(42)

#
# config
#

#parser = argparse.ArgumentParser()

#parser.add_argument('--train-dir', action='store', dest='train_dir',
#                    help='train directory location', required=True)

#args = parser.parse_args()

train_dir = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTest/'

#
# load topics
#
hi = []

with jsonlines.open(os.path.join(train_dir, 'test_org.json')) as topics:
    for topic in topics:
       hi.append(topic)

print(hi[1])
print(hi[1].get('guid'))
print(hi[1].get('c_paras'))
print(hi[1].get('label'))

print(len(topic_list))

with open(os.path.join(train_dir, 'english_topics_train.txt'), 'w') as topics:
    for topic in topic_list[:1000]:
        topics.write('{}\n'.format(topic))

with open(os.path.join(train_dir, 'english_topics_val.txt'), 'w') as topics:
    for topic in topic_list[1000:]:
        topics.write('{}\n'.format(topic))

