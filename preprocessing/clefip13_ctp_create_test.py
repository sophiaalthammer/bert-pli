import os
import csv
import argparse
import random
import re
from xml.etree import ElementTree as ET
random.seed(42)


#
# creates balanced training input file for finetuning ebrt with 1 or 0 labels
#


#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--train-dir', action='store', dest='train_dir',
                    help='training file directory location', required=True)
parser.add_argument('--output-dir', action='store', dest='output_dir',
                    help='training output file location for train.tsv', required=True)
parser.add_argument('--corpus-dir', action='store', dest='corpus_dir',
                    help='training output file location for train.tsv', required=True)

args = parser.parse_args()

#train_dir = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2013_claims_to_passage/clef-ip-2013-clms-psg-TEST'
#output_dir = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/2013_claims_to_passage/clef-ip-2013-clms-psg-TEST'
#corpus_dir = '/mnt/c/Users/sophi/Documents/phd/data/clef-ip/corpus_small'

#
# load topics
#

with open(os.path.join(args.train_dir, 'topics.txt'), 'r') as file:
    topics = file.read().splitlines()

topic_id = []
for num_pos in range(0, len(topics), 5):
    topic_id.append(topics[num_pos].split('>')[1].split('<')[0])

topic_file = []
for num_pos in range(1, len(topics), 5):
    topic_file.append(topics[num_pos].split('>')[1].split('<')[0])

claims = []
for num_pos in range(3, len(topics), 5):
    claims.append(topics[num_pos].split('>')[1].split('<')[0])

assert len(topic_id) == len(topic_file) == len(claims)

print('Number of topics are {}'.format(topic_id))

#
# load the xml files of the topics and their text, filter them for english
#

def read_in_patent_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    abstract_text = []
    # read in abstract text
    for abstract in root.iter('abstract'):
        if abstract.attrib['lang'] == 'EN':
            for text in abstract:
                try:
                    abstract_text.append(re.sub('\n', '',text.text))
                except:
                    pass

    # read in claims
    claims_text = {}
    for claims in root.iter('claims'):
        if claims.attrib['lang'] == 'EN':
            for claim in claims:
                for text in claim:
                    try:
                        claims_text.update({claim.attrib['num']: re.sub('\n', '',text.text)})
                    except:
                        pass

    # read in descriptions
    desc_text = {}
    for descriptions in root.iter('description'):
        if descriptions.attrib['lang'] == 'EN':
            for desc in descriptions:
                try:
                    desc_text.update({desc.attrib['num'].lstrip('0'): re.sub('\n', '',desc.text)})
                except:
                    pass

    #text = [x for x in abstract_text if x is not None]
    #text.extend([x for x in claims_text.values() if x is not None])
    #text.extend([x for x in desc_text.values() if x is not None])
    return abstract_text, claims_text, desc_text


topic_texts = []
for num_pos in range(len(topic_file)):
    file_path = os.path.join(args.train_dir, 'tfiles', topic_file[num_pos])
    abstract_text, claims_text, desc_text = read_in_patent_xml(file_path)
    topic_claims = claims[num_pos].split('/patent-document')
    print(topic_claims)
    topic_claims = [re.findall(r'\d+', x)[0] for x in topic_claims if x is not '' and len(re.findall(r'\d+', x)) > 0]
    text = ''
    for claim in topic_claims:
        # only works if the claim is in english and it exists
        try:
            text = text + ' ' + claims_text.get(claim)
        except:
            pass
    topic_texts.append(text)

# if topic has no text, the text is not english
assert len(topic_texts) == len(topic_file)

english_topics = {}
for num_pos in range(len(topic_file)):
    if topic_texts[num_pos]:
        english_topics.update({topic_id[num_pos]:topic_texts[num_pos]})

print('Number of english topics is {}'.format(len(english_topics)))

# load candidates from qrels, load claims frmo corpus, only consider them if they are english

samples_pos = []
samples_neg = []

with open(os.path.join(args.train_dir, 'qrels.txt'), 'r') as reader:
    qrels = reader.read().splitlines()

num_pos= 0
num_neg= 0

with open(os.path.join(args.output_dir, 'test.tsv'), 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')

    for qrel in qrels:
        try:
            qrel = qrel.split(',')
            #qrel = ['PSG1', 'EP-1000000-A1', '/patent-document/claims/claim[2]']
            topic_id = qrel[0]
            doc_id = qrel[1]
            genre = 'claim' if 'claims' in qrel[2] else 'description' if 'description' in qrel[2] else 'abstract'
            claim_id = re.findall(r'\d+', qrel[2])[0]

            doc_name = doc_id.split('-')
            if doc_name[0] == 'EP':
                file_path = os.path.join(args.corpus_dir, doc_name[0], '00000{}'.format(doc_name[1][0]),
                                         doc_name[1][1:3], doc_name[1][3:5], doc_name[1][5:7], '{}.xml'.format(doc_id))
            else:
                file_path = os.path.join(args.corpus_dir, doc_name[0], '00{}'.format(doc_name[1][0:4]),
                                         doc_name[1][4:6], doc_name[1][6:8], doc_name[1][8:10], '{}.xml'.format(doc_id))

            doc_abstract, doc_claims, doc_description = read_in_patent_xml(file_path)

            #print(doc_abstract)
            #print(doc_claims)
            #print(doc_description)

            try:
                if genre == 'claim':
                    doc_text = doc_claims.get(claim_id)
                elif genre == 'description':
                    doc_text = doc_description.get(claim_id)
                else:
                    doc_text = doc_abstract[0]
                if doc_text:
                    samples_pos.append([1, topic_id, '{}_{}_{}'.format(doc_id, genre, claim_id), english_topics.get(topic_id), doc_text])
                    num_pos = num_pos + 1
                    # append negative samples
                    for neg_id in doc_claims.keys():
                        if neg_id != claim_id:
                            samples_neg.append(
                                [0, topic_id, '{}_{}_{}'.format(doc_id, 'claim', neg_id), english_topics.get(topic_id), doc_claims.get(neg_id)])
                            num_neg = num_neg + 1
                    for neg_id in doc_description.keys():
                        if neg_id != claim_id:
                            num_neg = num_neg + 1
                            samples_neg.append(
                                [0, topic_id, '{}_{}_{}'.format(doc_id, 'description', neg_id), english_topics.get(topic_id), doc_description.get(neg_id)])
                    if genre != 'abstract':
                        num_neg= num_neg + 1
                        samples_neg.append(
                            [0, topic_id, '{}_{}_{}'.format(doc_id, 'abstract', '0'), english_topics.get(topic_id),
                             doc_abstract[0]])
            except:
                print('{} is not an english document and therefore wont be considered'.format(doc_id))
        except:
            print('{} has no claim id in qrels therefore not considered'.format(doc_id))

    samples_neg_less = random.sample(samples_neg, len(samples_pos)*5)

    samples_neg_less.extend(samples_pos)
    # important: shuffle the train data
    random.shuffle(samples_neg_less)

    print('number of samples {}'.format(len(samples_neg_less)))
    print('positive samples {}'.format(num_pos))
    print('negative samples {}'.format(num_neg))
    print('actually length of negatives {}'.format(len(samples_neg_less)))

    # write text in test.tsv file
    for sample in samples_neg_less:
        tsv_writer.writerow(sample)