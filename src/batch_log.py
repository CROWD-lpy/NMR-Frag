"""逐batch CSV；即时刷新，续训追加，通过session_id区分重新启动的进程。"""
import csv
import os
from datetime import datetime
from pathlib import Path

LOSS_FIELDS = ('total', 'add', 'connect', 'rank')
AUX_LOSS_FIELDS = ('presence', 'count', 'align')
COLUMNS = ('timestamp', 'session_id', 'stage', 'split', 'epoch', 'pass_id', 'batch',
           'global_step', 'batch_size', *LOSS_FIELDS, 'lr', 'grad_norm', 'batch_seconds',
           'elapsed_seconds', 'fragment_repr', 'freeze_chem', 'connect_weight', 'rank_weight')


class BatchLog:
    def __init__(self, path, every=1, auxiliary=False):
        if every < 0:
            raise ValueError('batch-log-every必须大于等于0')
        self.every, self.file, self.writer = every, None, None
        columns = COLUMNS + AUX_LOSS_FIELDS if auxiliary else COLUMNS
        self.session_id = datetime.now().astimezone().strftime('%Y%m%dT%H%M%S.%f') + f'-{os.getpid()}'
        if not every:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        has_header = path.exists() and path.stat().st_size > 0
        if has_header:
            with path.open(encoding='utf-8', newline='') as f:
                if next(csv.reader(f), []) != list(columns):
                    raise ValueError(f'{path}表头不匹配，请使用新的run-dir')
        self.file = path.open('a', encoding='utf-8', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        if not has_header:
            self.writer.writeheader(); self.file.flush()

    def write(self, **row):
        if self.writer is None or (row['batch'] != 1 and row['batch'] % self.every):
            return
        self.writer.writerow({'timestamp': datetime.now().astimezone().isoformat(timespec='seconds'),
                              'session_id': self.session_id, **row})
        self.file.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.file is not None:
            self.file.close()
