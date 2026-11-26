"""04：从PubChem预训练共享余弦Add，再切换NMRexp谱图条件微调。"""
import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from data import (batches, open_data, sample_tasks, augment, condition, eligible,
                  SAMPLING_ID, EVAL_SEED)
from model import Model, Config, MODEL_SHAPE_FIELDS, NMR_AUX_WEIGHTS
from runtime import MODEL_FORMAT, load_checkpoint, save_checkpoint, restore_optimizer, optimizer_param_names
from hard_data import HARD_DATA_VERSION, sample_candidate_graphs
from chem import digest
from batch_log import BatchLog, LOSS_FIELDS, AUX_LOSS_FIELDS
from training_policy import resolve_defaults, make_lr_schedule, lr_ratio, EARLY_TASKS
from search import (add_search_arguments, add_spectrum_arguments, search_options, resolved_options,
                    predict_record, TRAIN_SEARCH, LEGACY_TRAIN_SEARCH, SPECTRUM_DEFAULTS)
from metrics import evaluate_one, mean_metrics
from rdkit import rdBase
from training_guard import capture_rng, restore_rng, clip_gradients, save_failure

rdBase.DisableLog("rdApp.warning")


def generate_valid(model, root, stage, n, options, spectrum):
    totals, count = Counter(), 0
    for rows in batches(root, stage, 'valid', size=1, shuffle=True):
        row = rows[0]
        if stage != 'pubchem' and not eligible(row, spectrum['mode'], spectrum['cohort']):
            continue
        _, candidates = predict_record(model, row, stage, options, spectrum)
        totals.update(evaluate_one(row['smiles'], candidates))
        count += 1
        if count >= n:
            break
    return mean_metrics(count, totals)


def resolve_shape(args, config, loaded):
    """新run使用默认值；加载时继承形状，拒绝悄悄忽略不同的命令行设置。"""
    for name in MODEL_SHAPE_FIELDS:
        value = getattr(args, name)
        if loaded and value is not None and value != getattr(config, name):
            raise ValueError(f'--{name.replace("_", "-")}与检查点不同；改变层数/维度需要从PubChem重训')
        if value is not None:
            setattr(config, name, value)
        setattr(args, name, getattr(config, name))
    config.__post_init__()
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, help='processed dataset; --resume inherits the checkpoint path')
    p.add_argument('--stage', choices=('pubchem', 'nmrexp'), required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    init = p.add_mutually_exclusive_group()
    init.add_argument('--init-from', type=Path)
    init.add_argument('--resume', type=Path)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--tasks-per-record', type=int, default=2)
    p.add_argument('--lr', type=float, default=None, help='新训练默认2e-4；续训默认继承基础学习率')
    p.add_argument('--lr-decay-scale', type=float, default=None,
                   help='余弦衰减时间倍率，越大下降越慢；新NMRexp默认1.5，PubChem为1，续训继承')
    defaults = Config()
    for name in MODEL_SHAPE_FIELDS:
        p.add_argument('--'+name.replace('_', '-'), type=int, default=None,
                       help=f'新run默认{getattr(defaults, name)}；加载时继承检查点')
    p.add_argument('--rank-weight', type=float, default=None, help='默认继承检查点；新模型默认0.3')
    p.add_argument('--fragment-repr', choices=('descriptor', 'gnn', 'hybrid'), default=None,
                   help='ADD片段表示；默认继承检查点，新模型为descriptor；三种表示均支持解冻')
    p.add_argument('--freeze-chem', action=argparse.BooleanOptionalAction, default=None,
                   help='新run默认解冻；加载时继承检查点，可显式冻结或解冻')
    p.add_argument('--fragment-cache-batch-size', type=int, default=512,
                   help='片段库分块编码大小；冻结时建缓存，解冻时每batch保留梯度重算')
    p.add_argument('--bootstrap', action='store_true',
                   help='fit a seed NMRexp model on step-03 local negatives for offline mining')
    p.add_argument('--hard-negatives', type=int, default=None, help='NMRexp每条记录每步抽取的困难图上限；新run默认4，续训继承')
    p.add_argument('--align-temperature', type=float, default=None, help='NMRexp对比Align温度；默认0.1，继承检查点')
    p.add_argument('--augment-warmup', type=int, default=None,
                   help='增强概率线性增加的epoch数；新训练默认3，0从第一轮使用完整概率；续训继承')
    p.add_argument('--modality-drop', type=float, default=None, help='H+C变单谱：新NMRexp默认0.1，PubChem为0.3，续训继承')
    p.add_argument('--formula-drop', type=float, default=None, help='新NMRexp默认0.1，PubChem为0.25，续训继承')
    p.add_argument('--field-drop', type=float, default=None, help='积分/裂分/J删减：新NMRexp默认0.05，PubChem为0.1，续训继承')
    p.add_argument('--peak-drop', type=float, default=None, help='默认0，续训继承')
    p.add_argument('--metadata', action=argparse.BooleanOptionalAction, default=None, help='溶剂/频率输入；新run关闭，续训继承')
    p.add_argument('--amp', choices=('none', 'bf16'), default=None, help='新run默认none，续训继承')
    p.add_argument('--stable-add', action=argparse.BooleanOptionalAction, default=None,
                   help='规范Add条件memory和共享状态q；新训练开启，resume继承；旧权重可显式开启')
    p.add_argument('--grad-abort-norm', type=float, default=None,
                   help='裁剪前梯度范数超过阈值则保存现场并停止；默认1000，resume继承，0关闭有限值阈值')
    p.add_argument('--eval-valid', type=int, default=None, help='每轮自由生成样本数；0关闭')
    add_search_arguments(p, 'eval_')
    add_spectrum_arguments(p, 'eval_')
    p.add_argument('--limit-train', type=int, default=0)
    p.add_argument('--limit-valid', type=int, default=2000, help='teacher-forcing验证样本；0全部')
    p.add_argument('--log-every', type=int, default=500)
    p.add_argument('--batch-log-every', type=int, default=100, help='CSV记录每N个batch及每轮首个batch；1全部，0关闭')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    if a.fragment_cache_batch_size < 1 or a.batch_log_every < 0:
        raise ValueError('fragment-cache-batch-size必须>0，batch-log-every必须>=0')
    if a.rank_weight is not None and a.rank_weight < 0:
        raise ValueError('rank-weight必须>=0')
    if min(a.epochs, a.batch_size, a.tasks_per_record) < 1 or min(a.limit_train, a.limit_valid) < 0:
        raise ValueError('epochs/batch-size/tasks-per-record必须>0，数据limit必须>=0')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    ck = None
    default_data = Path('data/processed/overall_3000')
    if a.stage == 'nmrexp' and not a.bootstrap:
        default_data = Path('data/processed/overall_3000_hard_candidates')
    if a.resume or a.init_from:
        data_path = a.data if a.resume else (a.data or default_data)
        model, ck, root, info = load_checkpoint(a.resume or a.init_from, a.device, data_path)
        cat = model.cat
        if a.resume and ck['stage'] != a.stage:
            raise ValueError('跨阶段请用--init-from')
        if a.stage == 'pubchem' and ck['stage'] != 'pubchem':
            raise ValueError('PubChem训练请使用PubChem检查点，不能反向切换NMRexp模型')
        resolve_shape(a, model.config, loaded=True)
    else:
        root, info, cat = open_data(a.data or default_data)
        config = resolve_shape(a, Config(rank_weight=.3 if a.rank_weight is None else a.rank_weight,
                                        max_heavy=info['input_manifest']['max_heavy']), loaded=False)
        model = Model(cat, config).to(a.device)
    a.data = root
    merged = info.get('hard_candidates')
    if merged is not None and merged.get('version') != HARD_DATA_VERSION:
        raise ValueError('Unsupported hard-candidate dataset version')
    candidate_mode = None
    if a.stage == 'nmrexp':
        candidate_mode = 'local' if merged is None else 'per_batch'
    run = a.run_dir
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'config.json').exists() and not a.resume:
        raise FileExistsError('run-dir已有训练配置；使用--resume或新目录')
    resolve_defaults(a, ck)
    saved_args = ck.get('train_args', {}) if a.resume else {}
    if a.resume and saved_args.get('candidate_mode') == 'local':
        a.bootstrap = True
    if a.bootstrap and candidate_mode != 'local':
        p.error('--bootstrap requires --stage nmrexp and an unmerged step-03 dataset')
    if candidate_mode == 'local' and not a.bootstrap:
        p.error('NMRexp training requires step-08 merged data; use --bootstrap to train a seed on step-03 data')
    default_search = LEGACY_TRAIN_SEARCH if a.resume else TRAIN_SEARCH
    options = search_options(a, default_search, 'eval_', saved_args)
    if a.eval_valid is None:
        a.eval_valid = saved_args.get('eval_valid', 100)
    if a.eval_valid < 0:
        raise ValueError('--eval-valid must be nonnegative')
    if a.grad_abort_norm is None:
        a.grad_abort_norm = saved_args.get('grad_abort_norm', 1000.)
    if not math.isfinite(a.grad_abort_norm) or a.grad_abort_norm < 0:
        raise ValueError('grad-abort-norm必须为有限非负数')
    if a.amp is None:
        a.amp = saved_args.get('amp', 'none')
    if a.metadata is None:
        a.metadata = saved_args.get('metadata', False)
    spectrum = resolved_options(a, dict(SPECTRUM_DEFAULTS, metadata=a.metadata), 'eval_', saved_args)
    for name, value in {**options, **spectrum}.items():
        setattr(a, 'eval_' + name, value)
    if not math.isfinite(a.lr) or not math.isfinite(a.lr_decay_scale) or min(a.lr, a.lr_decay_scale) <= 0:
        raise ValueError('lr和lr-decay-scale必须为有限正数')
    for prob in (a.modality_drop, a.formula_drop, a.field_drop, a.peak_drop):
        if not 0 <= prob <= 1:
            raise ValueError('删减概率必须在[0,1]')
    if a.stage == 'nmrexp' and not a.resume:
        model.enable_auxiliary(align=ck is not None)
    if a.align_temperature is not None:
        model.config.align_temperature = a.align_temperature
    a.align_temperature = model.config.align_temperature
    if not math.isfinite(a.align_temperature) or a.align_temperature <= 0:
        raise ValueError('align-temperature必须为有限正数')
    if a.hard_negatives is None:
        a.hard_negatives = ck.get('train_args', {}).get('hard_negatives', 4) if a.resume else 4
    if a.hard_negatives < 1:
        raise ValueError('hard-negatives必须>0')
    connect_empty = .5 if a.stage == 'nmrexp' else .3
    merged_sha = digest(root/'manifest.json') if a.stage == 'nmrexp' else None
    if a.resume and a.stage == 'nmrexp' and ck.get('train_args', {}).get('merged_manifest_sha256') != merged_sha:
        raise ValueError('续训融合数据发生变化；请使用原数据，或--init-from开始新run')
    auxiliary = a.stage == 'nmrexp' and model.config.auxiliary_tasks
    loss_fields = LOSS_FIELDS + AUX_LOSS_FIELDS if auxiliary else LOSS_FIELDS
    old_options = (model.config.fragment_repr, model.config.freeze_chem, model.config.rank_weight,
                   model.config.stable_add)
    if a.stable_add is None:
        a.stable_add = model.config.stable_add if a.resume else True
    model.config.stable_add = a.stable_add
    mode = a.fragment_repr or model.config.fragment_repr
    frozen = a.freeze_chem
    if frozen is None:
        frozen = model.config.freeze_chem
    if frozen and ck is None:
        raise ValueError('不能从随机权重冻结化学GNN；从零PubChem请使用--no-freeze-chem')
    model.configure_fragments(mode, frozen)
    if a.rank_weight is not None:
        model.config.rank_weight = a.rank_weight
    a.fragment_repr, a.freeze_chem, a.rank_weight = mode, frozen, model.config.rank_weight
    a.hidden, a.heads = model.config.hidden, model.config.heads
    if a.augment_warmup is None:
        a.augment_warmup = ck.get('train_args', {}).get('augment_warmup', 0) if a.resume else 3
    if a.augment_warmup < 0:
        raise ValueError('augment-warmup必须>=0')
    model.fragment_cache_batch_size = a.fragment_cache_batch_size
    # 包含冻结参数，续训时可保留它们的优化器状态。
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    start, best, step = 1, None, 0
    resume_cursor = ck.get('training_cursor') if a.resume else None
    if a.resume:
        if not ck.get('resume_safe', True):
            raise ValueError('故障检查点的参数或优化器非有限；请从更早的last.pt/best.pt恢复')
        restore_optimizer(opt, model, ck)
        start, best, step = ck['epoch']+1, ck['best'], ck['step']
        restore_rng(ck['rng'], model.device)
        if old_options != (mode, frozen, model.config.rank_weight, model.config.stable_add) or not (run/'best.pt').exists():
            best = None  # 新功能/新目录的best独立选择，历史权重仍从检查点继承。
        previous_search = {k: saved_args.get('eval_' + k, v) for k, v in LEGACY_TRAIN_SEARCH.items()}
        previous_spectrum = {k: saved_args.get('eval_' + k, v) for k, v in
                             dict(SPECTRUM_DEFAULTS, metadata=saved_args.get('metadata', False)).items()}
        if (options != previous_search or spectrum != previous_spectrum
                or a.eval_valid != saved_args.get('eval_valid', 100)):
            best = None
        if a.stage == 'nmrexp' and any(ck.get('train_args', {}).get(k) != getattr(a, k)
                                       for k in ('hard_negatives', 'align_temperature')):
            best = None
    if frozen:
        print(f'冻结共享化学GNN，分批建立{len(cat)}个片段的全局/原子缓存…', flush=True)
        model.build_fragment_cache()
    elif mode in ('gnn', 'hybrid'):
        print(f'解冻共享化学GNN：每batch分块编码{len(cat)}个片段，Add/存在性/Connect共用；Rank使用同一GNN', flush=True)
    print(f'model_format={MODEL_FORMAT} fragment_repr={mode} freeze_chem={frozen} batch_log={run / "batch_metrics.csv" if a.batch_log_every else "off"}', flush=True)
    print('模型配置：'+' '.join(f'{name}={getattr(model.config, name)}' for name in MODEL_SHAPE_FIELDS), flush=True)
    print('Add：两阶段共享候选读取条件/集合、余弦+证据点积全词表评分、CE监督', flush=True)
    print(f'Add尺度控制 stable_add={a.stable_add}；梯度裁剪=1，异常停止阈值={a.grad_abort_norm:g}（0关闭）', flush=True)
    if ck and old_options[-1] != a.stable_add:
        print('Add尺度控制设置已改变：旧权重可严格加载，但即时预测与旧评分不等价；best将重新选择。', flush=True)
    if auxiliary:
        print(f'NMRexp辅助：remaining_presence=0.2 count=0.1 align={NMR_AUX_WEIGHTS["align"] if model.config.graph_alignment else 0}; '
              f'空集合10%、完整10%、中间80%均匀长度；drop(H/C)={a.modality_drop:g} drop(F)={a.formula_drop:g} '
              f'lr_decay_scale={a.lr_decay_scale:g}', flush=True)
    if a.stage == 'nmrexp':
        print(f'候选片段读取谱条件：hidden={model.config.candidate_hidden} heads={model.config.candidate_heads}；'
              f'剩余Presence参与Add软评分；Count保持总片段数监督', flush=True)
        print(f'Add：候选读取已选集合+全局条件；余弦主评分+证据点积，CE监督；'
              f'layers={model.config.candidate_layers}', flush=True)
        print(f'逐batch困难候选：每记录最多{a.hard_negatives}个；Align/Rank共享候选和增强条件；'
              f'temperature={a.align_temperature:g}；Connect空连接/随机保留=50%/50%', flush=True)
        if merged is None:
            print('NMRexp bootstrap: using local bond-rewiring negatives from step 03.', flush=True)
        else:
            print(f'融合覆盖：{merged.get("counts", {})}', flush=True)
        if not model.config.graph_alignment:
            print('从零训练未建立图教师，Align为0；启用Align请从已训练检查点--init-from', flush=True)
    source = {'pubchem': 'PubChem', 'nmrexp': 'NMRexp'}[a.stage]
    train_n = info['reference_rows'].get(source+'/train', 0)
    if not train_n:
        raise ValueError('该阶段没有训练数据')
    train_n = min(train_n, a.limit_train) if a.limit_train else train_n
    steps_total = max(1, math.ceil(train_n/a.batch_size)*a.tasks_per_record*a.epochs)
    schedule = make_lr_schedule(steps_total, a.lr_decay_scale, ck if a.resume else None, step)
    def amp():
        return torch.autocast(device_type=model.device.type, dtype=torch.bfloat16) if a.amp == 'bf16' else nullcontext()
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}

    if a.stage == 'pubchem':
        settings['pubchem_sampling'] = dict(
            algorithm=SAMPLING_ID, eval_seed=EVAL_SEED
        )

        if a.resume and (
                ck.get('train_args', {}).get('pubchem_sampling')
                != settings['pubchem_sampling']
        ):
            if ck.get('training_cursor'):
                raise ValueError(
                    '采样顺序已改变，请从完成一个epoch的last.pt或best.pt续训，'
                    '不要使用旧的批次中途游标'
                )
            best = None

    settings['model'] = model.checkpoint_config()
    settings['merged_manifest_sha256'] = merged_sha
    settings['candidate_mode'] = candidate_mode
    settings['connect_empty_probability'] = connect_empty
    settings['auxiliary_weights'] = NMR_AUX_WEIGHTS if auxiliary else None
    settings['task_sampling'] = EARLY_TASKS if auxiliary else 'uniform'
    settings['presence_target'] = 'remaining' if auxiliary else None
    settings['add_loss'] = 'ce'
    if resume_cursor:
        if resume_cursor['epoch'] != start or model.checkpoint_config() != vars(Config(**ck['model_config'])):
            raise ValueError('同一epoch故障续训必须保持模型配置和游标一致')
        for name in ('batch_size', 'tasks_per_record', 'limit_train', 'seed', 'amp', 'lr',
                     'lr_decay_scale', 'hard_negatives', 'augment_warmup', 'metadata', 'epochs',
                     'modality_drop', 'formula_drop', 'field_drop', 'peak_drop'):
            if settings[name] != ck['train_args'][name]:
                raise ValueError(f'同一epoch故障续训不能改变{name}')
    (run / 'config.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2))
    metrics_path = run / 'metrics.csv'
    columns = ['epoch', 'train_total', 'train_add', 'train_connect', 'train_rank', 'valid_loss', 'valid_n', 'hit1', 'hit10', 'oracle_pool']
    if auxiliary:
        columns += [f'{split}_{name}' for split in ('train', 'valid') for name in AUX_LOSS_FIELDS]
    if metrics_path.exists() and metrics_path.stat().st_size:
        with metrics_path.open(newline='') as existing:
            if next(csv.reader(existing), []) != columns:
                raise ValueError('metrics.csv表头不匹配；旧run开启辅助任务请用--init-from和新run-dir')
    with metrics_path.open('a', newline='') as f, BatchLog(run/'batch_metrics.csv', a.batch_log_every, auxiliary) as batch_log:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not f.tell():
            writer.writeheader()
        started = time.monotonic()
        for epoch in range(start, a.epochs+1):
            model.train(); totals = dict.fromkeys(loss_fields, 0.); count = batch_number = 0
            hard_sum, hard_updates, hard_rows, hard_graphs, empty_rows = 0., 0, 0, 0, 0
            completed_batches = 0
            if resume_cursor:
                totals, count = dict(resume_cursor['totals']), resume_cursor['count']
                completed_batches = resume_cursor['completed_batches']
                hard_sum, hard_updates, hard_rows, hard_graphs, empty_rows = resume_cursor['hard_stats']
            augmentation = (min(1., (epoch-1)/max(1, a.augment_warmup-1))
                            if a.augment_warmup > 1 else 1.)
            for pass_id in range(a.tasks_per_record):
                task_epoch = (epoch-1)*a.tasks_per_record+pass_id+1
                for rows in batches(root, a.stage, 'train', a.batch_size, task_epoch, a.seed, True,
                                    a.limit_train, reference_only=True, include_candidates=candidate_mode == 'per_batch'):
                    batch_started = time.monotonic(); batch_number += 1
                    if batch_number <= completed_batches:
                        continue
                    inputs = augment(rows, task_epoch, a.seed, a.modality_drop*augmentation,
                                     a.formula_drop*augmentation, a.field_drop*augmentation,
                                     a.peak_drop*augmentation, a.metadata)
                    tasks = sample_tasks(rows, task_epoch, a.seed, early=auxiliary, connect_empty=connect_empty)
                    candidates = (sample_candidate_graphs(rows, task_epoch, a.seed, a.hard_negatives, local=candidate_mode == 'local')
                                  if a.stage == 'nmrexp' else None)
                    ratio = lr_ratio(step, schedule)
                    for group in opt.param_groups:
                        group['lr'] = a.lr*ratio
                    opt.zero_grad(set_to_none=True)
                    rng = capture_rng(model.device)
                    with amp():
                        losses = model.loss(rows, tasks, inputs, structure=a.stage=='pubchem', candidate_graphs=candidates)
                    try:
                        if not torch.isfinite(losses['total']):
                            raise FloatingPointError('loss非有限值')
                        losses['total'].backward()
                        grad_norm = clip_gradients(model, 1., a.grad_abort_norm or None)
                    except FloatingPointError as error:
                        cursor = dict(epoch=epoch, completed_batches=batch_number-1,
                            totals=dict(totals), count=count,
                            hard_stats=[hard_sum, hard_updates, hard_rows, hard_graphs, empty_rows])
                        state = dict(data=str(root), stage=a.stage, epoch=epoch-1, step=step,
                            best=best, train_args=settings, lr_schedule=schedule,
                            rng=rng, training_cursor=cursor)
                        batch = dict(rows=rows, tasks=tasks, inputs=inputs, candidates=candidates,
                            losses={k: float(v.detach()) for k, v in losses.items()})
                        save_failure(run, model, opt, state, batch, error)
                        raise
                    opt.step(); step += 1
                    if candidates is not None:
                        used = sum(bool(pool) for pool in candidates)
                        hard_rows += used; empty_rows += len(rows)-used
                        hard_graphs += sum(map(len, candidates))
                        if used:
                            hard_sum += float(losses['rank'].detach()); hard_updates += 1
                    count += len(rows)
                    values = dict(zip(loss_fields, torch.stack([losses[k].detach().float() for k in loss_fields]).cpu().tolist()))
                    for k, value in values.items():
                        totals[k] += value*len(rows)
                    batch_log.write(stage=a.stage, split='train', epoch=epoch, pass_id=pass_id+1,
                                    batch=batch_number, global_step=step, batch_size=len(rows), **values,
                                    lr=opt.param_groups[0]['lr'], grad_norm=float(grad_norm),
                                    batch_seconds=time.monotonic()-batch_started, elapsed_seconds=time.monotonic()-started,
                                    fragment_repr=mode, freeze_chem=frozen,
                                    connect_weight=model.config.connect_weight, rank_weight=model.config.rank_weight)
                    if a.log_every and step % a.log_every == 0:
                        parts = ' '.join(f'{k}={totals[k]/count:.3f}' for k in loss_fields if k != 'total')
                        print(f'epoch={epoch} step={step} loss={totals["total"]/count:.3f} [{parts}] lr={opt.param_groups[0]["lr"]:.3g}', flush=True)
            resume_cursor = None
            model.eval(); vtotal, vn = 0., 0
            vaux = dict.fromkeys(AUX_LOSS_FIELDS, 0.)
            with torch.no_grad():
                for valid_batch, rows in enumerate(batches(root, a.stage, 'valid', a.batch_size, limit=a.limit_valid,
                                                          reference_only=True, include_candidates=candidate_mode == 'per_batch'), 1):
                    batch_started = time.monotonic()
                    inputs = [condition(r, metadata=a.metadata) for r in rows]
                    candidates = (sample_candidate_graphs(rows, 0, a.seed, a.hard_negatives, local=candidate_mode == 'local')
                                  if a.stage == 'nmrexp' else None)
                    with amp():
                        losses = model.loss(rows, sample_tasks(rows, connect_empty=connect_empty), inputs,
                                             structure=a.stage=='pubchem', candidate_graphs=candidates)
                    values = dict(zip(loss_fields, torch.stack([losses[k].detach().float() for k in loss_fields]).cpu().tolist()))
                    vtotal += values['total']*len(rows); vn += len(rows)
                    if auxiliary:
                        for name in AUX_LOSS_FIELDS:
                            vaux[name] += values[name] * len(rows)
                    batch_log.write(stage=a.stage, split='valid', epoch=epoch, pass_id=0,
                                    batch=valid_batch, global_step=step, batch_size=len(rows), **values,
                                    lr=opt.param_groups[0]['lr'], grad_norm='',
                                    batch_seconds=time.monotonic()-batch_started, elapsed_seconds=time.monotonic()-started,
                                    fragment_repr=mode, freeze_chem=frozen,
                                    connect_weight=model.config.connect_weight, rank_weight=model.config.rank_weight)
            if not vn:
                raise ValueError('该阶段没有可验证标签')
            valid_loss = vtotal/vn
            model.eval()  # 清除混合精度验证产生的缓存，自由生成统一用float32。
            hit = generate_valid(model, root, a.stage, a.eval_valid, options, spectrum) if a.eval_valid else None
            key = (hit['hit1'], hit['hit10'], -valid_loss) if hit and hit['n'] else (-valid_loss,)
            improved = best is None or tuple(key) > tuple(best)
            if improved:
                best = key
            state = dict(data=str(root), stage=a.stage, epoch=epoch, step=step, best=best, optimizer=opt.state_dict(),
                         optimizer_param_names=optimizer_param_names(model, opt),
                         train_args=settings, hard_train=dict(mode=candidate_mode,
                         updates=hard_updates, mean=hard_sum/max(1, hard_updates), rows=hard_rows,
                         candidates=hard_graphs, empty_rows=empty_rows),
                         lr_schedule=schedule,
                         rng=capture_rng(model.device))
            save_checkpoint(run/'last.pt', model, **state)
            if improved:
                save_checkpoint(run/'best.pt', model, **state)
            metrics = {'epoch': epoch, **{'train_'+k: totals[k]/count for k in totals}, 'valid_loss': valid_loss,
                       'valid_n': hit['n'] if hit else 0, **{k: hit[k] if hit else '' for k in ('hit1', 'hit10', 'oracle_pool')}}
            if auxiliary:
                metrics.update({f'valid_{k}': value/vn for k, value in vaux.items()})
            writer.writerow(metrics); f.flush()
            msg = f'epoch={epoch} loss={totals["total"]/count:.3f} [add={totals["add"]/count:.3f} connect={totals["connect"]/count:.3f} rank={totals["rank"]/count:.3f}] valid_loss={valid_loss:.3f}'
            if hit:
                msg += f' n={hit["n"]} Hit1={hit["hit1"]:.2%} Hit10={hit["hit10"]:.2%} Pool={hit["oracle_pool"]:.2%}'
            if hard_updates:
                msg += f' hard_rank={hard_sum/hard_updates:.3f} hard_updates={hard_updates}'
            if auxiliary:
                msg += ' ' + ' '.join(f'{k}={totals[k]/count:.3f}' for k in AUX_LOSS_FIELDS)
            print(msg + f' time={(time.monotonic()-started)/60:.1f}m', flush=True)


if __name__ == '__main__':
    main()
