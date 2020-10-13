import os
import csv
import argparse
import jsonlines
import re
from xml.etree import ElementTree as ET
import random
random.seed(42)

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--train-dir', action='store', dest='train_dir',
                    help='training file directory location', required=True)
parser.add_argument('--train-topics', action='store', dest='train_topics',
                    help='training topics.xml file', required=True)
parser.add_argument('--corpus-dir', action='store', dest='corpus_dir',
                    help='corpus directory', required=True)
parser.add_argument('--test-dir', action='store', dest='test_dir',
                    help='test file directory location', required=True)
parser.add_argument('--test-topics', action='store', dest='test_topics',
                    help='test topics.xml file', required=True)


args = parser.parse_args()

#train_topics = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTraining/topics.xml'
#train_dir = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip-2011_PACTraining/files'
#corpus_dir = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/corpus'
#test_topics = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip_2011_PACTest/PAC_topics.xml'
#test_dir = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2011_prior_candidate_search/clef-ip_2011_PACTest/files'

#
# load gold labels as list
#


def load_topics_from_xml(xml_path: str):

    with open(xml_path) as fh:
        root = ET.fromstring('<?xml version="1.0"?><topic>{}</topic>'.
                                 format(''.join(re.findall('<num.*?</num>', fh.read(), re.S)))
                                 )
    topics = []
    for child in root:
        topics.append(child.text)
    print('{} topics found in {}'.format(len(topics), xml_path))
    return topics


topics_train = load_topics_from_xml(args.train_topics)
topics_test = load_topics_from_xml(args.test_topics)


# topics only from EP collection not from WO

#
# now filter topics for onyl English ones (abstract, claim and description need to be English)
# for that read in the file from the corpus and then filter for lang='EN'
#


# read in files from train_dir:

#
# load directory structure
#

def get_topics(train_dir):
    list_dir = [x for x in os.walk(train_dir)]

    english_topics = []
    english_topics_claim = []
    for topic in list_dir[0][2]:
    #topic = 'EP-1222910-A2.xml'
        tree = ET.parse(os.path.join(train_dir, topic))
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

        #print(abs_lang.keys())

        if 'EN' in abs_lang.keys() and 'EN' in claim_lang.keys() and 'EN' in desc_lang.keys():
            english_topics.append(topic)
        if 'EN' in claim_lang.keys():
            english_topics_claim.append(topic)
    return english_topics, english_topics_claim, abs_lang, claim_lang, desc_lang


def save_topics(topics: list, train_dir: str):
    with open (os.path.join(train_dir, 'english_topics.txt'), 'w+') as file:
        for topic in topics:
            file.write('{}\n'.format(topic))

english_topics, english_topics_claim, abs_lang, claim_lang, desc_lang = get_topics(args.train_dir)

print('number of only english topics: {}'.format(len(english_topics))) #100
print('number of topics with english claim: {}'.format(len(english_topics_claim))) #100 # also english claims: 100

save_topics(english_topics, args.train_dir)

# same for test
english_topics, english_topics_claim, abs_lang, claim_lang, desc_lang = get_topics(args.test_dir)

print('number of only english topics: {}'.format(len(english_topics))) #1351
print('number of topics with english claim: {}'.format(len(english_topics_claim))) #100 # also english claims: 1351

save_topics(english_topics, args.test_dir)

# how to read in files from the corpus

for topic in topics_train:
    #topic = topics_train[0]
    file_number = topic.split('-')[1]

    try:
        if file_number.startswith('0'):
            file_path = os.path.join(args.corpus_dir, 'EP', '000000', file_number[1:3], file_number[3:5], file_number[5:7], '{}.xml'.format(topic))
        else:
            file_path = os.path.join(args.corpus_dir, 'EP', '000001', file_number[1:3], file_number[3:5], file_number[5:7], '{}.xml'.format(topic))

        #print(file_path)
        #file_path = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/corpus/EP/000001/00/00/00/EP-1000000-A1.xml'

        tree = ET.parse(file_path)
        root = tree.getroot()

        # read in abstract text
        for abstract in root.iter('abstract'):
            if abstract.attrib['lang'] == 'EN':
                abstract_text = []
                for text in abstract:
                    abstract_text.append(text.text)

        # read in claims
        claims_text = {}
        for claims in root.iter('claims'):
            if claims.attrib['lang'] == 'EN':
                for claim in claims:
                    for text in claim:
                        claims_text.update({claim.attrib['num']: text.text})

        # read in descriptions
        desc_text = {}
        for descriptions in root.iter('description'):
            if descriptions.attrib['lang'] == 'EN':
                for desc in descriptions:
                    desc_text.update({desc.attrib['num']: desc.text})


        english_topics = []
        if abstract.attrib['lang'] == 'EN' and claims.attrib['lang'] == 'EN' and descriptions.attrib['lang'] == 'EN':
            english_topics.append(topic)
    except:
        print('file {} not found in corpus'.format(topic))



