import csv
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

from model import observables


def wilson(successes, total):
    z = norm.ppf(0.975)
    probability = successes / total
    denominator = 1 + z*z/total
    center = (probability + z*z/(2*total)) / denominator
    radius = z * np.sqrt(probability*(1-probability)/total + z*z/(4*total*total)) / denominator
    return center-radius, center+radius


def write_rows(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, config):
    runs, summaries, probabilities = [], [], []
    for group, activation, width in itertools.product(config['groups'], config['activations'], config['widths']):
        saved = np.load(root/'checkpoints'/f'{group}_{activation}_width{width}.npz')
        theta, stopped = saved['theta'], saved['stopped']
        output = observables(theta, activation, group, config)
        assert np.isfinite(theta).all()
        assert (output['mass'].sum(axis=1) <= 1 + 1e-10).all()
        assert len(theta) == len(config['weight_decay'])*config['seeds']['count']
        key_bits = output['key_mask'][:, 0].astype(int) + 2*output['key_mask'][:, 1].astype(int)
        key_names = ['empty',output['names'][0],output['names'][1],'+'.join(output['names'])]
        losses = output['cross_entropy'] + saved['penalty'] * np.sum(theta**2,axis=(1,2,3))
        for i in range(len(theta)):
            runs.append({
                'group':group,'activation':activation,'width':width,'weight_decay':saved['penalty'][i],
                'seed':saved['seeds'][i],'updates':int(stopped[i]) if stopped[i] else int(saved['last_step']),
                'converged':bool(stopped[i]),'cross_entropy':output['cross_entropy'][i],
                'total_loss':losses[i],'accuracy':output['accuracy'][i],
                'key_set':key_names[key_bits[i]],'spectral_count':output['spectral_count'][i],
                'first_class':output['names'][0],'second_class':output['names'][1],
                'first_class_mass':output['mass'][i,0],'second_class_mass':output['mass'][i,1],
                'first_class_multiplicity':output['multiplicity'][i,0],
                'second_class_multiplicity':output['multiplicity'][i,1],
                'all_neurons_same_class':bool(output['same_class'][i]),
                'centered_logit_norm':output['centered_logit_norm'][i],
            })
        for penalty in config['weight_decay']:
            selected = saved['penalty'] == penalty
            count = int(selected.sum())
            np.testing.assert_array_equal(saved['seeds'][selected], np.arange(config['seeds']['first'],config['seeds']['first']+config['seeds']['count']))
            identifying = dict(group=group,activation=activation,width=width,weight_decay=penalty)
            same = int(output['same_class'][selected].sum())
            low, high = wilson(same,count)
            summaries.append({
                **identifying,'n_seeds':count,'n_converged':int((stopped[selected]>0).sum()),
                'n_nonconverged':int((stopped[selected]==0).sum()),
                'n_perfect_classification':int((output['accuracy'][selected]==1).sum()),
                'mean_accuracy':output['accuracy'][selected].mean(),
                'mean_cross_entropy':output['cross_entropy'][selected].mean(),
                'mean_total_loss':losses[selected].mean(),
                'mean_spectral_count':output['spectral_count'][selected].mean(),
                'mean_character_mass':output['mass'][selected].sum(axis=1).mean(),
                'mean_first_class_multiplicity':output['multiplicity'][selected,0].mean(),
                'mean_second_class_multiplicity':output['multiplicity'][selected,1].mean(),
                'n_all_neurons_same_class':same,'same_class_probability':same/count,
                'same_class_wilson_lower':low,'same_class_wilson_upper':high,
            })
            for key in range(len(key_names)):
                n_key = int((key_bits[selected]==key).sum()); lower,upper = wilson(n_key,count)
                probabilities.append({**identifying,'key_set':key_names[key],'n_key':n_key,'n_seeds':count,'probability':n_key/count,'wilson_lower':lower,'wilson_upper':upper})
    write_rows(root/'runs.csv',runs)
    write_rows(root/'summary.csv',summaries)
    write_rows(root/'key_set_probabilities.csv',probabilities)
    print(json.dumps({'configurations':len(summaries),'runs':len(runs),'key_probabilities':len(probabilities)}))


if __name__ == '__main__':
    analyze(Path('results'),json.loads(Path('experiment.json').read_text()))
