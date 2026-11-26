"""非有限梯度的安全裁剪、故障现场保存；不跳过训练样本。"""
import json
import math
import warnings

import torch


def capture_rng(device):
    state = {'torch': torch.get_rng_state()}
    if device.type == 'cuda':
        state['cuda_current'] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    if device.type == 'cuda':
        if 'cuda_current' in state:
            torch.cuda.set_rng_state(state['cuda_current'], device)


@torch.no_grad()
def clip_gradients(model, max_norm, abort_norm=None):
    """先检查原始范数再裁剪；有限但异常大的梯度也可在更新前中止。"""
    if math.isnan(max_norm) or max_norm <= 0:
        raise ValueError('max_norm必须>0')
    if abort_norm is not None and (not math.isfinite(abort_norm) or abort_norm <= 0):
        raise ValueError('abort_norm必须为有限正数或None')
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        return 0.
    overflow = False
    try:
        # inf只测范数，不提前缩放梯度；停止时保留原始梯度供诊断。
        value = float(torch.nn.utils.clip_grad_norm_(
            parameters, math.inf, error_if_nonfinite=True))
    except RuntimeError as error:
        if 'non-finite' not in str(error):
            raise
        norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(p.grad, dtype=torch.float64) for p in parameters]))
        value = float(norm)
        if not math.isfinite(value):
            raise FloatingPointError('存在NaN/Inf梯度，已阻止optimizer.step()') from error
        overflow = True
    if abort_norm is not None and value > abort_norm:
        raise FloatingPointError(
            f'原始梯度范数={value:.6g}超过停止阈值={abort_norm:.6g}；已阻止optimizer.step()')
    scale = min(1., max_norm / (value + 1e-6))
    if scale < 1.:
        for p in parameters:
            p.grad.mul_(scale)
    if overflow and scale < 1.:
        warnings.warn(f'float32梯度范数溢出；float64范数={value:.6g}，已安全裁剪', RuntimeWarning)
    return value


@torch.no_grad()
def gradient_diagnostics(model):
    """只在停止/重放时统计；前10个参数定位梯度集中位置，不代表最初致因层。"""
    stats, norms = [], []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        norm = float(torch.linalg.vector_norm(grad, dtype=torch.float64))
        norms.append(norm)
        stats.append(dict(parameter=name, norm=norm,
                          max_abs=float(grad.abs().max()) if grad.numel() else 0.,
                          nonfinite=int((~torch.isfinite(grad)).sum())))
    stats.sort(key=lambda item: (item['nonfinite'] > 0,
                                item['norm'] if math.isfinite(item['norm']) else math.inf), reverse=True)
    total = math.hypot(*norms)
    stats = stats[:10]
    for item in stats:
        for key in ('norm', 'max_abs'):
            if not math.isfinite(item[key]):
                item[key] = str(item[key])
    return dict(grad_norm=total if math.isfinite(total) else str(total), top_gradients=stats)


def save_failure(run, model, optimizer, state, batch, error):
    """异常发生在optimizer.step之前；保留权重、优化器、批次和前向前RNG。"""
    from runtime import save_checkpoint, optimizer_param_names
    bad_gradients = [name for name, p in model.named_parameters()
                     if p.grad is not None and not torch.isfinite(p.grad).all()]
    bad_parameters = [name for name, p in model.named_parameters()
                      if not torch.isfinite(p).all()]
    bad_optimizer = any(isinstance(value, torch.Tensor) and not torch.isfinite(value).all()
                        for entry in optimizer.state.values() for value in entry.values())
    report = dict(error=str(error), epoch=state['training_cursor']['epoch'],
        completed_step=state['step'], failed_batch=state['training_cursor']['completed_batches']+1,
        losses={k: v if math.isfinite(v) else str(v) for k, v in batch['losses'].items()},
        bad_gradients=bad_gradients, bad_parameters=bad_parameters,
        optimizer_nonfinite=bool(bad_optimizer), row_ids=[r['row_id'] for r in batch['rows']],
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        device=str(model.device), amp=state['train_args']['amp'],
        stable_add=model.config.stable_add,
        grad_abort_norm=state['train_args'].get('grad_abort_norm'),
        **gradient_diagnostics(model))
    # 只确认参数/优化器有限；不保证更早的异常更新没有损害模型。
    resume_safe = not bad_parameters and not bad_optimizer
    save_checkpoint(run/'interrupted.pt', model, **state,
        optimizer=optimizer.state_dict(), optimizer_param_names=optimizer_param_names(model, optimizer),
        failure_batch=batch, failure_report=report, resume_safe=resume_safe)
    (run/'nonfinite.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'异常现场已保存：{run / "interrupted.pt"}；诊断信息：{run / "nonfinite.json"}', flush=True)
