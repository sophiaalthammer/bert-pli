import os
import argparse
import random
import jsonlines
import json
import re
from xml.etree import ElementTree as ET
random.seed(42)

#
# config
#
#parser = argparse.ArgumentParser()

#parser.add_argument('--corpus-dir', action='store', dest='corpus_dir',
#                    help='corpus directory location', required=True)
#parser.add_argument('--json-dir', action='store', dest='json_dir',
#                    help='json target directory location', required=True)
#

#args = parser.parse_args()

corpus_dir = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/corpus_small'
json_dir = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/corpus_json'

#
# load directory structure
#

#english_topics = []
for root, dirs, files in os.walk(corpus_dir):
    # chooses the first document of the files, because this is always the relevant one we are referring to
    if files:
        file_path = os.path.join(root, files[0])
        print(file_path)
        tree = ET.parse(file_path)
        root = tree.getroot()

        abstract_text = []
        # read in abstract text
        for abstract in root.iter('abstract'):
            if abstract.attrib['lang'] == 'EN':
                for text in abstract:
                    try:
                        abstract_text.append(text.text)
                    except:
                        pass

        # read in claims
        claims_text = {}
        for claims in root.iter('claims'):
            if claims.attrib['lang'] == 'EN':
                for claim in claims:
                    for text in claim:
                        try:
                            claims_text.update({claim.attrib['num']: text.text})
                        except:
                            pass

        # read in descriptions
        desc_text = {}
        for descriptions in root.iter('description'):
            if descriptions.attrib['lang'] == 'EN':
                for desc in descriptions:
                    try:
                        desc_text.update({desc.attrib['num']: desc.text})
                    except:
                        pass

        # only english and non empty topics! at least abstract, claim there
        if abstract_text is not None and claims_text is not None and \
                abstract.attrib['lang'] == 'EN' and \
                claims.attrib['lang'] == 'EN' and \
                descriptions.attrib['lang'] == 'EN':
            #english_topics.append(file_path)

            json_file_path = file_path.split('/')[-1].split('.')[0]

            with open(os.path.join(json_dir, '{}.json'.format(json_file_path)), 'w+') as json_file:
                json.dump({'id': json_file_path,
                           'contents': ' '.join([x for x in abstract_text]) +
                                      ' '.join([x for x in claims_text.values()]) +
                                      ' '.join([x for x in desc_text.values() if desc_text.values() is not None])}, json_file)


