import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from model import adam_step, group_data


def initial_weights(config, group, width):
    group_number = config['groups'].index(group)
    order = len(group_data(group)[0])
    seeds = range(config['seeds']['first'], config['seeds']['first'] + config['seeds']['count'])
    weights = np.stack([
        np.random.default_rng([config['seeds']['base'], group_number, width, seed]).normal(
            scale=order ** -0.5, size=(3, width, order),
        ) for seed in seeds
    ])
    return np.tile(weights, (len(config['weight_decay']), 1, 1, 1))


def run(args):
    config = json.loads(args.config.read_text())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    args.resume.mkdir(parents=True, exist_ok=True)
    table = group_data(args.group)[0]
    targets = torch.eye(len(table), dtype=torch.float64, device=device)[torch.as_tensor(table, device=device)]
    penalty = torch.as_tensor(np.repeat(config['weight_decay'], config['seeds']['count']), device=device, dtype=torch.float64)
    started = time.perf_counter()
    for width in config['widths']:
        stem = f'{args.group}_{args.activation}_width{width}'
        final = args.output / f'{stem}.npz'
        if final.exists():
            print(f'{stem}: completed checkpoint present', flush=True)
            continue
        saved = args.resume / f'{stem}.pt'
        theta = torch.as_tensor(initial_weights(config, args.group, width), dtype=torch.float64, device=device)
        first = torch.zeros_like(theta)
        second = torch.zeros_like(theta)
        stable = torch.zeros(len(theta), dtype=torch.int64, device=device)
        stopped = torch.zeros_like(stable)
        start_step = 1
        if saved.exists():
            state = torch.load(saved, map_location=device, weights_only=True)
            theta, first, second = state['theta'], state['first'], state['second']
            stable, stopped = state['stable'], state['stopped']
            start_step = state['step'] + 1
        def update(theta, first, second, stable, stopped, step):
            new_theta, new_first, new_second, loss, loss_change, parameter_change = adam_step(
                theta, first, second, penalty, targets, step, args.activation, config,
            )
            active = stopped == 0
            stable_now = (loss_change < config['relative_tolerance']) & (parameter_change < config['relative_tolerance'])
            new_stable = torch.where(stable_now, stable + 1, 0)
            stable = torch.where(active, new_stable, stable)
            stopped = torch.where(active & (stable >= config['stable_steps']), step.to(torch.int64), stopped)
            active = active[:, None, None, None]
            return (
                torch.where(active, new_theta, theta), torch.where(active, new_first, first),
                torch.where(active, new_second, second), stable, stopped,
            )
        compiled = torch.compile(update, fullgraph=True)
        step_tensor = torch.zeros((), dtype=torch.float64, device=device)
        for step in range(start_step, config['max_steps'] + 1):
            step_tensor.fill_(step)
            theta, first, second, stable, stopped = compiled(theta, first, second, stable, stopped, step_tensor)
            if step % 1000 == 0 or step == config['max_steps']:
                torch.cuda.synchronize(device)
                assert bool(torch.isfinite(theta).all())
                torch.save({
                    'theta': theta, 'first': first, 'second': second, 'stable': stable,
                    'stopped': stopped, 'step': step,
                }, saved)
                print(f'{stem}: step={step}, converged={int((stopped > 0).sum())}/{len(theta)}, seconds={time.perf_counter()-started:.1f}', flush=True)
                if bool((stopped > 0).all()):
                    break
        np.savez_compressed(
            final, theta=theta.cpu().numpy(), stopped=stopped.cpu().numpy(),
            stable=stable.cpu().numpy(), last_step=step,
            seeds=np.tile(np.arange(config['seeds']['first'], config['seeds']['first']+config['seeds']['count']), len(config['weight_decay'])),
            penalty=penalty.cpu().numpy(),
        )
        print(f'{stem}: saved {final.name}', flush=True)
    (args.output / f'{args.group}_{args.activation}_runtime.json').write_text(json.dumps({
        'seconds': time.perf_counter() - started, 'torch': torch.__version__,
        'numpy': np.__version__, 'device': torch.cuda.get_device_name(device),
        'dtype': 'float64', 'config': config,
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=['C5', 'S3'], required=True)
    parser.add_argument('--activation', choices=['quadratic', 'relu'], required=True)
    parser.add_argument('--config', type=Path, default=Path('experiment.json'))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, default=Path('results/checkpoints'))
    parser.add_argument('--resume', type=Path, required=True)
    run(parser.parse_args())
