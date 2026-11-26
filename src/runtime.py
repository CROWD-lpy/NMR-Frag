"""仅加载本版检查点；模型参数严格匹配，不提供旧权重迁移。"""
from dataclasses import asdict
import torch
from data import open_data
from model import Model, Config

MODEL_FORMAT = 'v6-cosine-ce-1'


def load_checkpoint(path, device='cpu', data=None):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('model_format') != MODEL_FORMAT:
        raise ValueError(f'仅支持本版 {MODEL_FORMAT} 检查点；旧MLP评分参数不能直接resume，请从PubChem重新训练')
    root, info, cat = open_data(data or ck['data'])
    if ck['tokenizer_sha256'] != cat.tokenizer_sha256:
        raise ValueError('检查点与词表不匹配')
    config = Config(**ck['model_config'])
    if config.max_heavy != info['input_manifest']['max_heavy']:
        raise ValueError('数据max_heavy与检查点配置不一致')
    model = Model(cat, config).to(device)
    model.load_state_dict(ck['model'], strict=True)
    return model, ck, root, info


def optimizer_param_names(model, optimizer):
    names = {id(p): name for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def restore_optimizer(optimizer, model, checkpoint):
    """同版、同结构续训；不猜测或迁移缺失的优化器参数。"""
    names = optimizer_param_names(model, optimizer)
    if checkpoint['optimizer_param_names'] != names:
        raise ValueError('优化器参数名或顺序不一致；无法resume')
    state = checkpoint['optimizer']
    if len(state['param_groups']) != len(names):
        raise ValueError('优化器参数组不一致')
    parameters = dict(model.named_parameters())
    for group_names, group in zip(names, state['param_groups']):
        if len(group_names) != len(group['params']):
            raise ValueError('优化器参数数量不一致')
        for name, pid in zip(group_names, group['params']):
            for key, value in state['state'].get(pid, {}).items():
                if key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq') and value.shape != parameters[name].shape:
                    raise ValueError(f'优化器状态维度不一致：{name}/{key}')
    optimizer.load_state_dict(state)


def save_checkpoint(path, model, **kwargs):
    ck = dict(model_format=MODEL_FORMAT, model=model.state_dict(), model_config=asdict(model.config),
              tokenizer_sha256=model.cat.tokenizer_sha256, **kwargs)
    temp = path.with_suffix('.tmp')
    torch.save(ck, temp)
    temp.replace(path)
