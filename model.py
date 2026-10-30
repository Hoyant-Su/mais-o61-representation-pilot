import itertools

import numpy as np
import torch


def group_data(name):
    if name == 'C5':
        order = 5
        table = (np.arange(order)[:, None] + np.arange(order)[None, :]) % order
        inverse = (-np.arange(order)) % order
        characters = np.exp(2j * np.pi * np.arange(1, order)[:, None] * np.arange(order) / order)
        classes = [[0, 3], [1, 2]]
        names = ['frequency_1', 'frequency_2']
        difference = np.arange(order)[:, None] - np.arange(order)[None, :]
        projectors = np.stack([2 * np.cos(2 * np.pi * k * difference / order) / order for k in [1, 2]])
    else:
        elements = list(itertools.permutations(range(3)))
        index = {p: i for i, p in enumerate(elements)}
        order = len(elements)
        table = np.array([[index[tuple(p[q[k]] for k in range(3))] for q in elements] for p in elements])
        inverse = np.array([index[tuple(np.argsort(p))] for p in elements])
        signs = np.array([(-1) ** sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3)) for p in elements])
        standard = np.array([sum(i == p[i] for i in range(3)) - 1 for p in elements])
        characters = np.stack([signs, standard])
        classes = [[0], [1]]
        names = ['sign', 'standard']
        sign_projection = np.outer(signs, signs) / order
        projectors = np.stack([sign_projection, np.eye(order) - np.ones((order, order)) / order - sign_projection])
    product_inverse = table[table[:, :, None], inverse[None, None, :]]
    basis = characters[:, product_inverse]
    return table, basis, projectors, classes, names


def logits(theta, activation):
    pre = theta[:, 0, :, :, None] + theta[:, 1, :, None, :]
    hidden = pre.square() if activation == 'quadratic' else torch.relu(pre)
    return torch.einsum('riab,ric->rabc', hidden, theta[:, 2])


def loss_gradient(theta, penalty, targets, activation):
    pre = theta[:, 0, :, :, None] + theta[:, 1, :, None, :]
    hidden = pre.square() if activation == 'quadratic' else torch.relu(pre)
    values = torch.einsum('riab,ric->rabc', hidden, theta[:, 2])
    order = theta.shape[-1]
    ce = (torch.logsumexp(values, dim=-1) - (values * targets).sum(dim=-1)).mean(dim=(-1, -2))
    residual = (torch.softmax(values, dim=-1) - targets) / (order * order)
    gw = torch.einsum('riab,rabc->ric', hidden, residual)
    gh = torch.einsum('rabc,ric->riab', residual, theta[:, 2])
    derivative = 2 * pre if activation == 'quadratic' else (pre > 0).to(theta.dtype)
    gz = gh * derivative
    gradient = torch.stack([gz.sum(dim=-1), gz.sum(dim=-2), gw], dim=1)
    gradient = gradient + 2 * penalty[:, None, None, None] * theta
    loss = ce + penalty * theta.square().sum(dim=(1, 2, 3))
    return loss, gradient


def adam_step(theta, first, second, penalty, targets, step, activation, config):
    loss, gradient = loss_gradient(theta, penalty, targets, activation)
    first = config['beta1'] * first + (1 - config['beta1']) * gradient
    second = config['beta2'] * second + (1 - config['beta2']) * gradient.square()
    corrected_first = first / (1 - config['beta1'] ** step)
    corrected_second = second / (1 - config['beta2'] ** step)
    updated = theta - config['learning_rate'] * corrected_first / (corrected_second.sqrt() + config['adam_epsilon'])
    values = logits(updated, activation)
    ce = (torch.logsumexp(values, dim=-1) - (values * targets).sum(dim=-1)).mean(dim=(-1, -2))
    new_loss = ce + penalty * updated.square().sum(dim=(1, 2, 3))
    relative_loss = (new_loss - loss).abs() / loss.abs().clamp_min(1)
    relative_parameter = (updated - theta).square().sum(dim=(1, 2, 3)).sqrt() / theta.square().sum(dim=(1, 2, 3)).sqrt().clamp_min(1)
    return updated, first, second, new_loss, relative_loss, relative_parameter


def observables(theta, activation, group, config):
    table, basis, projectors, classes, names = group_data(group)
    pre = theta[:, 0, :, :, None] + theta[:, 1, :, None, :]
    hidden = pre ** 2 if activation == 'quadratic' else np.maximum(pre, 0)
    values = np.einsum('riab,ric->rabc', hidden, theta[:, 2])
    centered = values - values.mean(axis=-1, keepdims=True)
    norm = np.sqrt(np.sum(centered ** 2, axis=(1, 2, 3)))
    numerator = np.einsum('rabc,jabc->rj', centered, basis.conj())
    denominator = norm[:, None] * np.sqrt(np.sum(abs(basis) ** 2, axis=(1, 2, 3)))[None, :]
    beta = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    mass = np.stack([np.sum(abs(beta[:, cls]) ** 2, axis=1) for cls in classes], axis=-1)
    keys = np.sqrt(mass) >= config['key_threshold']
    total = mass.sum(axis=-1)
    sorted_mass = np.sort(mass, axis=-1)[:, ::-1]
    count = np.argmax(np.cumsum(sorted_mass, axis=-1) >= (1 - config['spectral_delta']) * total[:, None], axis=-1) + 1
    count[total == 0] = 0
    weights = theta - theta.mean(axis=-1, keepdims=True)
    energy = np.sum(weights ** 2, axis=(1, 3))
    projected = np.einsum('kgh,rtih->rktig', projectors, weights)
    projected_energy = np.sum(projected ** 2, axis=(2, 4))
    pure = (projected_energy >= (1 - config['purity_delta']) * energy[:, None, :]) & (energy[:, None, :] > 0)
    shifted = values - values.max(axis=-1, keepdims=True)
    ce = (np.log(np.exp(shifted).sum(axis=-1)) - np.take_along_axis(shifted, table[None, :, :, None], axis=-1)[..., 0]).mean(axis=(1, 2))
    return {
        'mass': mass, 'key_mask': keys, 'spectral_count': count,
        'multiplicity': pure.mean(axis=-1), 'same_class': pure.all(axis=-1).any(axis=-1),
        'accuracy': (values.argmax(axis=-1) == table).mean(axis=(1, 2)),
        'cross_entropy': ce, 'centered_logit_norm': norm, 'names': names,
    }
