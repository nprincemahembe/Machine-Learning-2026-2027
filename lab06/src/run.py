
import argparse
import csv
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
LAB = ROOT / 'lab06'
RESULTS = LAB / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

try:
    from lab03.src.models import CNN
except ImportError as exc:
    raise SystemExit('Cannot import CNN from lab03/src/models.py. Check the Lab 3 model filename/class.') from exc


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_datasets(seed):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.286,), (0.353,)),
    ])
    ds = datasets.FashionMNIST(str(ROOT / 'data'), train=True, download=True, transform=transform)
    train, val = random_split(ds, [55000, 5000], generator=torch.Generator().manual_seed(seed))
    return train, val


def loaders(train, val, batch, workers, seed):
    
    gen = torch.Generator().manual_seed(seed)
    opts = {'num_workers': workers, 'pin_memory': torch.cuda.is_available()}
    if workers:
        opts['persistent_workers'] = True
    return (
        DataLoader(train, batch_size=batch, shuffle=True, generator=gen, **opts),
        DataLoader(val, batch_size=256, shuffle=False, **opts),
    )


def optimizer_and_scheduler(model, optimizer_name, schedule, epochs, lr, momentum):
    if optimizer_name == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if schedule == 'step':
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
    elif schedule == 'cosine':
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', patience=2)
    return opt, sch


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def train_one_epoch(model, loader, opt, device, accum):
    model.train()
    opt.zero_grad(set_to_none=True)
    n = 0
    loss_total = 0.0
    start = time.perf_counter()
    # Use summed losses / actual group sample count to handle final partial group correctly.
    pending = 0
    for idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        loss_sum = F.cross_entropy(model(x), y, reduction='sum')
        loss_sum.backward()
        pending += len(y)
        n += len(y)
        loss_total += loss_sum.item()
        if (idx + 1) % accum == 0 or (idx + 1) == len(loader):
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(pending)
            opt.step()
            opt.zero_grad(set_to_none=True)
            pending = 0
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return loss_total / n, n / elapsed


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        predictions = model(x.to(device)).argmax(dim=1).cpu()
        correct += (predictions == y).sum().item()
        total += len(y)
    return correct / total


def run_experiment(cfg, train, val, device, tag, *, batch=None, workers=None,
                   epochs=None, optimizer=None, scheduler=None, accum=None,
                   resume=False, stop_after=None):
    import mlflow
    batch = batch if batch is not None else cfg['batch_size']
    workers = workers if workers is not None else cfg['num_workers']
    epochs = epochs if epochs is not None else cfg['epochs']
    optimizer = optimizer or cfg['optimizer']
    scheduler = scheduler or cfg['scheduler']
    accum = accum if accum is not None else cfg['accumulation_steps']
    lr = cfg['learning_rate'] if optimizer == 'adam' else 0.01
    seed_everything(cfg['seed'])
    dl, vl = loaders(train, val, batch, workers, cfg['seed'])
    model = CNN().to(device)
    opt, sch = optimizer_and_scheduler(model, optimizer, scheduler, epochs, lr, cfg['momentum'])
    checkpoint = RESULTS / 'checkpoints' / f'{tag}.pt'
    checkpoint.parent.mkdir(exist_ok=True)
    start_epoch, history = 0, []
    if resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f'Checkpoint missing: {checkpoint}')
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        sch.load_state_dict(state['scheduler'])
        start_epoch = state['epoch']
        history = state['history']
        torch.set_rng_state(state['torch_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        if device.type == 'cuda' and 'cuda_rng' in state:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'Resuming {tag} from epoch {start_epoch}')
    mlflow.set_tracking_uri((ROOT / 'mlruns').resolve().as_uri())
    mlflow.set_experiment('lab06_training')
    limit = min(epochs, stop_after) if stop_after else epochs
    with mlflow.start_run(run_name=tag):
        mlflow.log_params({'tag': tag, 'batch': batch, 'workers': workers,
                           'epochs': epochs, 'optimizer': optimizer, 'scheduler': scheduler,
                           'accumulation': accum, 'seed': cfg['seed'], 'device': str(device)})
        for epoch in range(start_epoch, limit):
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            loss, throughput = train_one_epoch(model, dl, opt, device, accum)
            acc = validate(model, vl, device)
            if scheduler == 'plateau':
                sch.step(acc)
            else:
                sch.step()
            peak = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 3) if device.type == 'cuda' else 'N/A (CPU)'
            row = {'epoch': epoch + 1, 'train_loss': round(loss, 6),
                   'val_accuracy': round(acc, 6), 'throughput_samples_s': round(throughput, 2),
                   'peak_cuda_mb': peak, 'learning_rate': opt.param_groups[0]['lr']}
            history.append(row)
            mlflow.log_metrics({'train_loss': loss, 'val_accuracy': acc,
                                'throughput_samples_s': throughput,
                                'learning_rate': opt.param_groups[0]['lr']}, step=epoch + 1)
            if device.type == 'cuda':
                mlflow.log_metric('peak_cuda_mb', float(peak), step=epoch + 1)
            torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                        'scheduler': sch.state_dict(), 'epoch': epoch + 1,
                        'history': history, 'torch_rng': torch.get_rng_state(),
                        'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                        **({'cuda_rng': torch.cuda.get_rng_state_all()} if device.type == 'cuda' else {})}, checkpoint)
            print(f'{tag}: epoch {epoch + 1}/{epochs} | accuracy {acc:.4f} | {throughput:.0f} samples/s')
        out = RESULTS / f'{tag}.csv'
        save_csv(out, history)
        mlflow.log_artifact(str(out))
        mlflow.log_artifact(str(checkpoint))
    return history



def profile_epoch(cfg, train, val, device):
    seed_everything(cfg['seed'])

    dl, _ = loaders(
        train, val, cfg['batch_size'], 0, cfg['seed']
    )

    model = CNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    durations = {
        'data_loading': 0.0,
        'forward': 0.0,
        'backward': 0.0,
        'optimizer': 0.0
    }

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # Measure all batches in one complete epoch.
    # Do not retain profiler events for the full epoch.
    iterator = iter(dl)
    start_epoch = time.perf_counter()

    for _ in range(len(dl)):
        start = time.perf_counter()
        x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        sync()
        durations['data_loading'] += time.perf_counter() - start

        opt.zero_grad(set_to_none=True)

        start = time.perf_counter()
        loss = F.cross_entropy(model(x), y)
        sync()
        durations['forward'] += time.perf_counter() - start

        start = time.perf_counter()
        loss.backward()
        sync()
        durations['backward'] += time.perf_counter() - start

        start = time.perf_counter()
        opt.step()
        sync()
        durations['optimizer'] += time.perf_counter() - start

    epoch_seconds = time.perf_counter() - start_epoch

    # Collect a small PyTorch profiler trace separately.
    # This avoids storing hundreds of batches of events.
    profile_batches = min(5, len(dl))
    profile_iterator = iter(dl)

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False
    ) as prof:

        for _ in range(profile_batches):
            x, y = next(profile_iterator)
            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)

            with record_function('stage_forward'):
                loss = F.cross_entropy(model(x), y)

            with record_function('stage_backward'):
                loss.backward()

            with record_function('stage_optimizer'):
                opt.step()

    (RESULTS / 'profile.txt').write_text(
        prof.key_averages().table(
            sort_by='self_cpu_time_total',
            row_limit=40
        ),
        encoding='utf-8'
    )

    prof.export_chrome_trace(
        str(RESULTS / 'profile.json')
    )

    total = sum(durations.values())

    rows = [
        {
            'stage': stage,
            'seconds': round(seconds, 4),
            'percent': round(
                100 * seconds / total, 2
            )
        }
        for stage, seconds in durations.items()
    ]

    save_csv(
        RESULTS / 'profile_breakdown.csv',
        rows
    )

    summary = [
        'Lab 6 Profiling Summary',
        '',
        f'Device: {device}',
        f'Full epoch time: {epoch_seconds:.4f} seconds',
        f'Batches measured: {len(dl)}',
        f'Batches traced with torch.profiler: {profile_batches}',
        '',
        'Full-epoch stage breakdown:'
    ]

    for row in rows:
        summary.append(
            f"{row['stage']}: "
            f"{row['seconds']:.4f} seconds "
            f"({row['percent']:.2f}%)"
        )

    summary.extend([
        '',
        'Note: The percentage breakdown uses the sum of',
        'the four measured training stages as denominator.',
        'Iterator setup, zero_grad and other overhead are',
        'not included in those percentages.',
        'The detailed torch.profiler trace uses only',
        'a small sample of batches to avoid memory errors.'
    ])

    (RESULTS / 'profile_summary.txt').write_text(
        '\n'.join(summary),
        encoding='utf-8'
    )

    print('\nPROFILING COMPLETED SUCCESSFULLY')
    print(f'Full epoch: {epoch_seconds:.2f} seconds')
    print(f'Batches measured: {len(dl)}')
    print(f'Batches profiled: {profile_batches}')

    for row in rows:
        print(
            f"{row['stage']}: "
            f"{row['percent']:.2f}%"
        )

    print('\nResults saved in:', RESULTS)



def plot_study(path, x_key, y_key, name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with path.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    xs = [float(r[x_key]) for r in rows]
    ys = [float(r[y_key]) for r in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel(x_key.replace('_', ' '))
    plt.ylabel(y_key.replace('_', ' '))
    plt.title(name)
    plt.tight_layout()
    plt.savefig(RESULTS / (path.stem + '.png'), dpi=150)
    plt.close()


def make_report(cfg, device):
    lines = ['# Lab 6: Training Pipelines and Profiling', '',
             f'Configuration: `lab06/config.yaml`; seed: {cfg["seed"]}; device: {device}.',
             'Model: `CNN` imported from Lab 3; dataset: Fashion-MNIST.', '',
             '## Evidence', '']
    for filename in ['profile_breakdown.csv', 'workers_study.csv', 'batch_study.csv',
                     'optimizer_study.csv', 'accumulation_study.csv', 'resume_demo.csv']:
        p = RESULTS / filename
        if p.exists():
            lines.append(f'- `{filename}` generated.')
        else:
            lines.append(f'- `{filename}` not yet generated; run its study.')
    lines.extend(['', '## Interpretation',
                  'The fastest data-loading worker count depends on hardware and Windows process overhead.',
                  'Batch size changes throughput and memory use; GPU peak memory is unavailable on CPU.',
                  'Compare optimizer and scheduler results under equal ten-epoch budgets.',
                  'Gradient accumulation approximates a larger effective batch; exact equivalence can be affected by batch-dependent layers.',
                  'Resume demonstration uses the saved model, optimizer, scheduler, epoch and RNG states.',
                  '', '## Project proposal',
                  'Handwritten digit recognition on MNIST with a small CNN. Target: at least 97% test accuracy,',
                  'single-image inference <= 50 ms, model size <= 5 MB. Planned optimization: quantization.',
                  'These are targets that must be measured, not established results.', ''])
    (LAB / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(LAB / 'config.yaml'))
    p.add_argument('--study', choices=['workers', 'batch', 'optimizers', 'accumulation', 'profile', 'resume'])
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    seed_everything(cfg['seed'])
    device = torch.device('cuda' if cfg.get('use_cuda', True) and torch.cuda.is_available() else 'cpu')
    print('Lab 6 | device:', device, '| study:', args.study or 'default')
    (RESULTS / 'versions.txt').write_text(
        f'Python: {sys.version}\nPyTorch: {torch.__version__}\nNumPy: {np.__version__}\nDevice: {device}\n', encoding='utf-8')
    train, val = make_datasets(cfg['seed'])
    if args.study == 'profile':
        profile_epoch(cfg, train, val, device)
    elif args.study == 'workers':
        rows = []
        for w in [0, 1, 2, 4, 8]:
            hist = run_experiment(cfg, train, val, device, f'workers_{w}', workers=w, epochs=1)
            rows.append({'num_workers': w, **hist[-1]})
        save_csv(RESULTS / 'workers_study.csv', rows)
        plot_study(RESULTS / 'workers_study.csv', 'num_workers', 'throughput_samples_s', 'Data loading worker study')
    elif args.study == 'batch':
        rows = []
        for b in [32, 64, 128, 256, 512]:
            hist = run_experiment(cfg, train, val, device, f'batch_{b}', batch=b, epochs=5)
            rows.append({'batch_size': b, **hist[-1]})
        save_csv(RESULTS / 'batch_study.csv', rows)
        plot_study(RESULTS / 'batch_study.csv', 'batch_size', 'throughput_samples_s', 'Batch-size throughput')
    elif args.study == 'optimizers':
        rows = []
        for opt in ['sgd', 'adam']:
            for sched in ['step', 'cosine', 'plateau']:
                hist = run_experiment(cfg, train, val, device, f'{opt}_{sched}', optimizer=opt, scheduler=sched, epochs=10)
                rows.append({'optimizer': opt, 'scheduler': sched, **hist[-1]})
        save_csv(RESULTS / 'optimizer_study.csv', rows)
    elif args.study == 'accumulation':
        rows = []
        for batch, accum in [(64, 4), (256, 1)]:
            hist = run_experiment(cfg, train, val, device, f'accum_b{batch}_a{accum}', batch=batch, accum=accum, epochs=5)
            rows.append({'batch_size': batch, 'accumulation': accum, 'effective_batch': batch * accum, **hist[-1]})
        save_csv(RESULTS / 'accumulation_study.csv', rows)
    elif args.study == 'resume':
        tag = 'resume_demo'
        ck = RESULTS / 'checkpoints' / f'{tag}.pt'
        if ck.exists():
            ck.unlink()  # Explicitly start a fresh interruption demonstration.
        run_experiment(cfg, train, val, device, tag, epochs=5, stop_after=3)
        hist = run_experiment(cfg, train, val, device, tag, epochs=5, resume=True)
        save_csv(RESULTS / 'resume_demo.csv', hist)
    else:
        run_experiment(cfg, train, val, device, 'default', resume=args.resume)
    make_report(cfg, device)
    print('Finished. Results:', RESULTS)
    print('MLflow runs:', ROOT / 'mlruns')
    print('Report:', LAB / 'report.md')


if __name__ == '__main__':
    # Required for DataLoader(num_workers > 0) on Windows.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
