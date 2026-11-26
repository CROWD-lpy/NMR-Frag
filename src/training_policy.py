"""训练默认值与可续训的慢余弦学习率；不改变旧PubChem的默认策略。"""
import math


LEGACY_DROPS = dict(modality_drop=.3, formula_drop=.25, field_drop=.1, peak_drop=0.)
CHF_DROPS = dict(modality_drop=.1, formula_drop=.1, field_drop=.05, peak_drop=0.)
EARLY_TASKS = dict(empty=.10, complete=.10, intermediate=.80, intermediate_length='uniform_1_to_N_minus_1')


def resolve_defaults(args, checkpoint):
    """续训继承已保存选项；新NMRexp用CHF优先，新PubChem沿用旧值。"""
    saved = checkpoint.get('train_args', {}) if args.resume else {}
    drops = CHF_DROPS if args.stage == 'nmrexp' and not args.resume else LEGACY_DROPS
    for name, default in drops.items():
        if getattr(args, name) is None:
            setattr(args, name, saved.get(name, default))
    if args.lr is None:
        args.lr = saved.get('lr', 2e-4)
    if args.lr_decay_scale is None:
        args.lr_decay_scale = saved.get('lr_decay_scale',
                                       1.5 if args.stage == 'nmrexp' and not args.resume else 1.)


def lr_ratio(step, schedule):
    """step为当前已完成步数；旧scale=1时与原04公式完全一致。"""
    warmup = schedule['warmup']
    if step < warmup:
        return min(1., (step + 1) / warmup)
    if 'anchor_step' in schedule:
        progress = schedule['anchor_progress'] + (
            (step - schedule['anchor_step']) / max(1, schedule['total_steps'] - schedule['anchor_step'])
            * (1. - schedule['anchor_progress']) / schedule['scale'])
    else:
        progress = (step - warmup) / max(1, schedule['total_steps'] - warmup) / schedule['scale']
    return .1 + .9 * (1. + math.cos(math.pi * min(1., max(0., progress)))) / 2.


def make_lr_schedule(total_steps, scale, checkpoint=None, step=0):
    schedule = dict(total_steps=total_steps, warmup=min(500, max(1, total_steps // 20)), scale=scale)
    previous = checkpoint.get('lr_schedule') if checkpoint else None
    if previous:
        if previous['total_steps'] == total_steps and previous['scale'] == scale:
            return dict(previous)
        schedule['warmup'] = previous['warmup']
        if step >= previous['warmup']:
            # 延长epochs/修改衰减速度时，以最后一次实际LR为锚点，避免突然回升。
            old_base = checkpoint['train_args']['lr']
            last_ratio = checkpoint['optimizer']['param_groups'][0]['lr'] / old_base
            cosine = max(-1., min(1., 2. * (last_ratio - .1) / .9 - 1.))
            schedule.update(anchor_step=step, anchor_progress=math.acos(cosine) / math.pi)
    return schedule
