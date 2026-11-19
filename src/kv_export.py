"""Opt-in, scheduler-thread KV observations; never substitutes GPU-wide usage."""
import functools
import json
import os
import time
import uuid
from pathlib import Path
from batch.ownership import identity


def geometry(pool):
    value = dict(layers=pool.layer_num, element_bytes=pool.store_dtype.itemsize)
    if hasattr(pool, 'kv_lora_rank'):
        value.update(kind='mla', kv_lora_rank=pool.kv_lora_rank,
                     qk_rope_head_dim=pool.qk_rope_head_dim)
    else:
        value.update(kind='mha', local_kv_heads=pool.head_num, head_dim=pool.head_dim)
    return value


def pool_values(pool, snapshot):
    """Page-one allocated slots exclude the reserve ledger and null slot."""
    if pool.page_size != 1:
        raise ValueError('Only page_size=1 has a proven block-to-token contract')
    shape = geometry(pool)
    width = (shape['local_kv_heads'] * shape['head_dim'] if shape['kind'] == 'mha'
             else shape['kv_lora_rank'] + shape['qk_rope_head_dim'])
    expected = width * shape['element_bytes']
    buffers = 2 if shape['kind'] == 'mha' else 1
    if (snapshot['block_size_bytes'] != expected or snapshot['num_layers'] != shape['layers']
            or snapshot['num_kv_buffers'] != buffers):
        raise ValueError('Observed pool geometry and block bytes disagree')
    active = snapshot['allocated_blocks'] - snapshot['reserved_blocks'] - int(snapshot['null_block_reserved'])
    if active < 0:
        raise ValueError('Invalid allocator slot accounting')
    return dict(active_tokens=active, physical_bytes=snapshot['total_mapped_bytes'],
                geometry=shape, capacity_tokens=pool.size)


def capture(scheduler):
    import torch
    from sglang.srt.distributed import get_tensor_model_parallel_rank
    from kvcached.integration.sglang import interfaces as kvi
    pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    runtime = kvi.observability_snapshot_dict()
    snapshots = kvi.kv_cache_pool_snapshot_dicts()
    if len(snapshots) != 1 or not runtime['initialized']:
        raise ValueError('Expected one initialized SGLang KV pool')
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
    return dict(checkpoint=scheduler.model_config.model_path, rank=get_tensor_model_parallel_rank(),
                pid=os.getpid(), process_identity=identity(os.getpid()), gpu='GPU-' + str(uuid.UUID(str(gpu).removeprefix('GPU-'))), raw_gpu_uuid=str(gpu), prefix_cache_enabled=not scheduler.disable_radix_cache,
                dp_attention=scheduler.enable_dp_attention, kvcached_enabled=runtime['initialized'],
                observed_epoch=time.time(), runtime=runtime, pool=snapshots[0],
                operations=kvi.kv_cache_pool_operation_snapshot_dicts(), **pool_values(pool, snapshots[0]))


def publish(directory, value):
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = directory / (str(os.getpid()) + '.json')
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(target)


def install():
    """Install only in the explicitly enabled independent launcher processes."""
    directory = os.environ.get('KVREPRO_KV_EXPORT_DIRECTORY')
    if not directory:
        return
    from sglang.srt.managers.scheduler import Scheduler
    original = Scheduler.process_input_requests
    if getattr(original, '_kvrepro_export', False):
        return
    @functools.wraps(original)
    def observed(self, requests):
        result = original(self, requests)
        if time.monotonic() >= getattr(self, '_kvrepro_next_sample', 0):
            self._kvrepro_next_sample = time.monotonic() + 1
            try:
                publish(directory, capture(self))
            except Exception as error:
                try:
                    publish(directory, dict(pid=os.getpid(), observed_epoch=time.time(), error=str(error)))
                except OSError:
                    pass  # Missing/stale output makes the external probe fail closed.
        return result
    observed._kvrepro_export = True
    Scheduler.process_input_requests = observed
