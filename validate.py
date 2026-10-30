import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from model import adam_step, group_data, loss_gradient, observables


def reference_loss(theta, penalty, table, activation):
    rows = []
    for a in range(len(table)):
        for b in range(len(table)):
            hidden = theta[0, :, a] + theta[1, :, b]
            hidden = hidden.square() if activation == 'quadratic' else functional.relu(hidden)
            rows.append(hidden @ theta[2])
    return functional.cross_entropy(torch.stack(rows), torch.as_tensor(table.ravel())) + penalty * theta.square().sum()


def character_network(frequency, order=5):
    angle = 2 * np.pi * frequency * np.arange(order) / order
    cosine, sine = np.cos(angle), np.sin(angle)
    terms = [(cosine, cosine, cosine), (sine, sine, -cosine), (sine, cosine, sine), (cosine, sine, sine)]
    neurons = []
    for left, right, output in terms:
        neurons.extend([np.stack([left, right, output / 4]), np.stack([left, -right, -output / 4])])
    return np.stack(neurons, axis=1)


def validate(config):
    torch.set_num_threads(1)
    rng = np.random.default_rng(861)
    for group in config['groups']:
        table, basis, projectors, classes, names = group_data(group)
        order = len(table)
        indices = np.arange(order)
        np.testing.assert_array_equal(table[0], indices)
        np.testing.assert_array_equal(table[:, 0], indices)
        for a in indices:
            for b in indices:
                for c in indices:
                    assert table[table[a, b], c] == table[a, table[b, c]]
        np.testing.assert_allclose(basis.reshape(len(basis), -1) @ basis.reshape(len(basis), -1).conj().T, order ** 3 * np.eye(len(basis)), atol=1e-12)
        np.testing.assert_allclose(projectors.sum(axis=0), np.eye(order) - np.ones((order, order))/order, atol=1e-12)
        for i, projection in enumerate(projectors):
            np.testing.assert_allclose(projection @ projection, projection, atol=1e-12)
            np.testing.assert_allclose(projection, projection.T, atol=1e-12)
            np.testing.assert_allclose(projection @ projectors[1-i], 0, atol=1e-12)
        targets = torch.eye(order, dtype=torch.float64)[torch.as_tensor(table)]
        for activation in config['activations']:
            theta = torch.tensor(rng.normal(size=(3, 3, 2, order)), dtype=torch.float64)
            penalty = torch.tensor(config['weight_decay'], dtype=torch.float64)
            losses, gradient = loss_gradient(theta, penalty, targets, activation)
            for index in range(len(theta)):
                single = theta[index].clone().requires_grad_()
                loss = reference_loss(single, penalty[index], table, activation)
                expected = torch.autograd.grad(loss, single)[0]
                torch.testing.assert_close(losses[index], loss, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(gradient[index], expected, rtol=1e-11, atol=1e-12)
            references = [torch.nn.Parameter(x.clone()) for x in theta]
            optimizers = [torch.optim.Adam([x], lr=config['learning_rate'], betas=(config['beta1'], config['beta2']), eps=config['adam_epsilon']) for x in references]
            first, second = torch.zeros_like(theta), torch.zeros_like(theta)
            for step in range(1, 51):
                theta, first, second, losses, _, _ = adam_step(theta, first, second, penalty, targets, step, activation, config)
                for index, (parameter, optimizer) in enumerate(zip(references, optimizers)):
                    optimizer.zero_grad()
                    reference_loss(parameter, penalty[index], table, activation).backward()
                    optimizer.step()
            torch.testing.assert_close(theta, torch.stack(references), rtol=1e-10, atol=1e-11)
            measured = observables(theta.numpy(), activation, group, config)
            for index, parameter in enumerate(references):
                expected = reference_loss(parameter, 0.0, table, activation).detach().item()
                np.testing.assert_allclose(measured['cross_entropy'][index], expected, rtol=1e-12, atol=1e-12)
            assert (measured['mass'].sum(axis=1) <= 1 + 1e-12).all()
        zero = observables(np.zeros((1, 3, 2, order)), 'quadratic', group, config)
        assert zero['spectral_count'][0] == 0
        assert not zero['key_mask'].any()
        assert not zero['multiplicity'].any()
        constant = observables(np.ones((1, 3, 2, order)), 'quadratic', group, config)
        assert not constant['multiplicity'].any()
    pure = character_network(1)[None]
    measured = observables(pure, 'quadratic', 'C5', config)
    np.testing.assert_allclose(measured['mass'], [[1, 0]], atol=1e-12)
    np.testing.assert_array_equal(measured['key_mask'], [[True, False]])
    np.testing.assert_array_equal(measured['spectral_count'], [1])
    np.testing.assert_allclose(measured['multiplicity'], [[1, 0]], atol=1e-12)
    second_frequency = character_network(2)
    second_frequency[2] *= 0.5
    mixed = observables(np.concatenate([pure[0], second_frequency], axis=1)[None], 'quadratic', 'C5', config)
    np.testing.assert_allclose(mixed['mass'], [[0.8, 0.2]], atol=1e-12)
    np.testing.assert_array_equal(mixed['spectral_count'], [2])
    transformed = observables(pure[..., (3*np.arange(5)) % 5], 'quadratic', 'C5', config)
    np.testing.assert_allclose(transformed['mass'], measured['mass'][:, ::-1], atol=1e-12)
    np.testing.assert_allclose(transformed['multiplicity'], measured['multiplicity'][:, ::-1], atol=1e-12)
    print('PASS: group laws, character orthogonality, projectors, independent gradients and Adam, pure/mixed/zero spectra, constant neurons, automorphism relabeling')


if __name__ == '__main__':
    validate(json.loads(Path('experiment.json').read_text()))
