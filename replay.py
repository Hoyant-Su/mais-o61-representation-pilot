import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from model import group_data
from train import initial_weights


def replay(group, activation, config):
    torch.set_num_threads(1)
    width, penalty, seed = 8, 0.0001, 17
    table = group_data(group)[0]
    index = config['weight_decay'].index(penalty) * config['seeds']['count'] + seed - config['seeds']['first']
    saved = np.load(f'results/checkpoints/{group}_{activation}_width{width}.npz')
    steps = int(saved['stopped'][index]) or int(saved['last_step'])
    theta = torch.nn.Parameter(torch.tensor(initial_weights(config, group, width)[index], dtype=torch.float64))
    target = torch.as_tensor(table.ravel())
    optimizer = torch.optim.Adam([theta], lr=config['learning_rate'], betas=(config['beta1'],config['beta2']), eps=config['adam_epsilon'])
    for step in range(steps):
        optimizer.zero_grad()
        pre = theta[0, :, :, None] + theta[1, :, None, :]
        hidden = pre.square() if activation=='quadratic' else functional.relu(pre)
        predictions = hidden.permute(1,2,0) @ theta[2]
        loss = functional.cross_entropy(predictions.reshape(-1,len(table)), target) + penalty*theta.square().sum()
        loss.backward()
        optimizer.step()
    observed = theta.detach().numpy()
    expected = saved['theta'][index]
    difference = np.linalg.norm(observed-expected)
    relative = difference/max(1,np.linalg.norm(expected))
    result = dict(group=group,activation=activation,width=width,weight_decay=penalty,seed=seed,updates=steps,relative_parameter_difference=relative,max_absolute_parameter_difference=float(np.max(abs(observed-expected))))
    Path(f'results/replay_{group}_{activation}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    assert relative < 1e-6


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--group',choices=['C5','S3'],required=True)
    parser.add_argument('--activation',choices=['quadratic','relu'],required=True)
    args=parser.parse_args()
    replay(args.group,args.activation,json.loads(Path('experiment.json').read_text()))
