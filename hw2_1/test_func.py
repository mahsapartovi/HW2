import sys
import torch
import pickle
import json
from compute_cuda_Mahsa import testing_dataset, test, MODELS, rnn_encoder, rnn_decoder, attention
from torch.utils.data import DataLoader
from bleu_evaluation import BLEU

device =0
fpath = './data/testing_data/feat'
model = torch.load('./SavedModel/model0.h5', map_location=lambda storage, loc: storage)
dataset = testing_dataset('{}'.format(sys.argv[1]))
test_loader = DataLoader(dataset, batch_size=32, shuffle=True)
with open('indextoword.pickle', 'rb') as handle:
    i2w = pickle.load(handle)
model = model.to(device)
sentences = test(test_loader, model, i2w)
with open(sys.argv[2], 'w') as f:
    for id, s in sentences:
        f.write('{},{}\n'.format(id, s))


test = json.load(open('./data/testing_label.json'))
output = sys.argv[2]
result = {}
with open(output,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption
bleu_score=[]
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']],captions,True))
    bleu_score.append(score_per_video[0])
average = sum(bleu_score) / len(bleu_score)
print("Average bleu score is " + str(average))