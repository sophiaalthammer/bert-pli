import os
import argparse
import random
from pyserini.search import SimpleSearcher
import jsonlines
from xml.etree import ElementTree as ET
random.seed(42)

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--index-dir', action='store', dest='index_dir',
                    help='train directory location', required=True)
parser.add_argument('--topic-dir', action='store', dest='topic_dir',
                    help='topic file location', required=True)

args = parser.parse_args()

#index_dir = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/corpus_small_index'
#topic_dir = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip_2011_PACTraining/'

#
# load topics
#

with open(os.path.join(args.topic_dir, 'failed_dirs.txt'), 'r') as topics:
    topic_list = topics.read().splitlines()

print(topic_list)

#
# load directory structure
#


with open(os.path.join(args.topic_dir, 'failed_dirs_50.txt'),'w') as failed_dir:
    for topic in topic_list:
        try:
            # read in query text
            tree = ET.parse(os.path.join(args.topic_dir, 'files', topic))
            root = tree.getroot()
            

            abs_lang = {}
            for abstract in root.iter('abstract'):
                for text in abstract:
                    abs_lang.update({abstract.attrib['lang']: text.text})

            claim_lang = {}
            for claims in root.iter('claims'):
                claim_dict = {}
                for claim in claims:
                    for text in claim:
                        try:
                            claim_dict.update({claim.attrib['num']: text.text})
                        except:
                            pass
                    claim_lang.update({claims.attrib['lang']: claim_dict})

            desc_lang = {}
            for descriptions in root.iter('description'):
                desc_dict = {}
                for desc in descriptions:
                    try:
                        desc_dict.update({desc.attrib['num']: desc.text})
                    except:
                        pass
                desc_lang.update({descriptions.attrib['lang']: desc_dict})

            text_abs = abs_lang.get('EN').strip().replace('\n', '')
            text_claim = ' '.join(list(filter(None, [text for text in claim_lang.get('EN').values()]))).strip()
            text_desc = ' '.join(list(filter(None, [text for text in desc_lang.get('EN').values()]))).strip()

            query_text = ' '.join((text_abs + text_claim + text_desc).split()[:250])
            #print(query_text)
            
            searcher = SimpleSearcher(args.index_dir)
            searcher.set_bm25(0.9, 0.4)

            hits = searcher.search(query_text, 50)

            # Print the first 50 hits:
            with open(os.path.join(args.topic_dir, 'bm25_top50','bm25_top50_{}.txt'.format(topic)),"w",encoding="utf8") as out_file:
                for i in range(0, 50):
                    out_file.write(f'{hits[i].docid:55}\n')
                    print(f'{hits[i].docid:55}')
        except:
            print('failed for {}'.format(topic))
            failed_dir.write('{}\n'.format(topic))
