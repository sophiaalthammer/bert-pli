import torch
checkpoint = '/mnt/c/Users/salthamm/Documents/phd/data/models/bert-pli-reprod/bertorg_lawrnn_gru.pkl'
checkpoint_out = '/mnt/c/Users/salthamm/Documents/phd/data/models/bert-pli-reprod/bertorg_lawrnn_gru2.pkl'

parameters = torch.load(checkpoint, map_location=torch.device('cpu'))
model_parameters = parameters['model']
print(model_parameters.keys())

model_parameters2 = {'module.rnn.weight_ih_l0':model_parameters.get('rnn.module.weight_ih_l0'),
                     'module.rnn.weight_hh_l0': model_parameters.get('rnn.module.weight_hh_l0'),
                     'module.rnn.bias_ih_l0': model_parameters.get('rnn.module.bias_ih_l0'),
                     'module.rnn.bias_hh_l0': model_parameters.get('rnn.module.bias_hh_l0'),
                     'module.fc_a.weight': model_parameters.get('fc_a.module.weight'),
                     'module.fc_a.bias': model_parameters.get('fc_a.module.bias'),
                     'module.fc_f.weight': model_parameters.get('fc_f.module.weight'),
                     'module.fc_f.bias': model_parameters.get('fc_f.module.bias')}
                     #'criterion.weight': model_parameters.get('criterion.weight')}

parameters['model'] = model_parameters2

torch.save(parameters, checkpoint_out)