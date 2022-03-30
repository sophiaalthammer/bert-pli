import torch
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--ck_in', action='store', dest='ck_in',
                    help='training file directory location', required=True)
parser.add_argument('--ck_out', action='store', dest='ck_out',
                    help='training file directory location', required=True)
args = parser.parse_args()

#run_before_poolout = '/mnt/c/Users/sophi/Documents/phd/data/coliee2022/task1/train/run_for_bertpli_top1_15_0.json'
#run_after_poolout = '/mnt/c/Users/sophi/Documents/phd/data/coliee2022/task1/bertpli/output/train/run_for_bertpli_top1_15_0.json'

checkpoint = args.ck_in
checkpoint_out = args.ck_out


#checkpoint = '/mnt/c/Users/salthamm/Documents/phd/data/models/bert-pli-reprod/bertorg_lawrnn_gru.pkl'
#checkpoint_out = '/mnt/c/Users/salthamm/Documents/phd/data/models/bert-pli-reprod/bertorg_lawrnn_gru2.pkl'

parameters = torch.load(checkpoint, map_location=torch.device('cpu'))
model_parameters = parameters['model']
print(model_parameters.keys())
print(model_parameters.get('rnn.weight_ih_l0'))
print(model_parameters.get('rnn.weight_hh_l0'))

#model_parameters2 = {'module.rnn.weight_ih_l0':model_parameters.get('rnn.weight_ih_l0'),
#                     'module.rnn.weight_hh_l0': model_parameters.get('rnn.weight_hh_l0'),
#                     'module.rnn.bias_ih_l0': model_parameters.get('rnn.bias_ih_l0'),
#                     'module.rnn.bias_hh_l0': model_parameters.get('rnn.bias_hh_l0'),
#                     'module.fc_a.weight': model_parameters.get('fc_a.weight'),
#                     'module.fc_a.bias': model_parameters.get('fc_a.bias'),
#                     'module.fc_f.weight': model_parameters.get('fc_f.weight'),
#                     'module.fc_f.bias': model_parameters.get('fc_f.bias'),
#                     'module.criterion.weight': model_parameters.get('criterion.weight')}

model_parameters2 = {'module.rnn.weight_ih_l0':model_parameters.get('rnn.module.weight_ih_l0'),
                     'module.rnn.weight_hh_l0': model_parameters.get('rnn.module.weight_hh_l0'),
                     'module.rnn.bias_ih_l0': model_parameters.get('rnn.module.bias_ih_l0'),
                     'module.rnn.bias_hh_l0': model_parameters.get('rnn.module.bias_hh_l0'),
                     'module.fc_a.weight': model_parameters.get('fc_a.module.weight'),
                     'module.fc_a.bias': model_parameters.get('fc_a.module.bias'),
                     'module.fc_f.weight': model_parameters.get('fc_f.module.weight'),
                     'module.fc_f.bias': model_parameters.get('fc_f.module.bias'),
                     'module.criterion.weight': model_parameters.get('criterion.weight')}

parameters['model'] = model_parameters2

torch.save(parameters, checkpoint_out)
