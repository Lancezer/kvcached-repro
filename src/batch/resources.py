"""独立服务/MPS生命周期。探针只读实际状态，失败不伪造就绪。"""
import json
import os
import subprocess
import time
from pathlib import Path
from cleanup_diagnostics import cleanup_diagnostic, cleanup_operation, emit_cleanup_diagnostic
from .health import healthy
from .ownership import SharedFault, identity, matches, terminate, session_members
from .storage import write
from .audit import kv_bytes
from . import ipc
from .ipc_journal import CreationJournal


class Resources:
    def __init__(self, config, row, directory):
        self.config, self.row, self.directory = config, row, directory
        self.processes, self.records, self.handles = [], [], []
        self.intent = None
        self.mps_env = {}
        self.server_pids = []
        self.services, self.service_death = [], []
        self.descendants = []
        self.client_pid = None
        self.allocations = []

    def persist(self):
        write(self.directory / 'resources.json', dict(records=self.records, intent=self.intent,
              mps_env=self.mps_env, server_pids=self.server_pids, descendants=self.descendants,
              allocations=self.allocations, ipc_evidence=[], cleaned=False))

    def launch(self, args, env, name):
        self.intent = dict(name=name, args=args, epoch=time.time(), owner=identity(os.getpid()))
        self.persist()
        log = (self.directory / (name + '.log')).open('x')
        self.handles.append(log)
        try:
            proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        except OSError:
            # Popen 已报告启动失败，服务程序没有执行。
            self.intent = None
            self.persist()
            raise
        self.processes.append(proc)
        if name == 'client':
            self.client_pid = proc.pid
        record = identity(proc.pid)
        if record is None:
            raise SharedFault('启动与登记间隙进程消失，需审计')
        self.records.append(dict(record, role=name))
        self.intent = None
        self.persist()
        return proc

    def mps_control(self, text):
        args = [self.config['mps_control']]
        result = subprocess.run(args, env=dict(os.environ, **self.mps_env), input=text + '\n',
                                text=True, capture_output=True, timeout=self.config['startup_seconds'])
        result.check_returncode()
        return result.stdout

    def start_mps(self):
        if not self.row['mps']:
            self.off_domain()
            return
        root = self.directory / 'mps'
        root.mkdir(mode=0o700)
        for name in ('pipe', 'log'):
            (root / name).mkdir(mode=0o700)
        self.mps_env = dict(CUDA_MPS_PIPE_DIRECTORY=str(root / 'pipe'),
                            CUDA_MPS_LOG_DIRECTORY=str(root / 'log'))
        controller = self.launch([self.config['mps_control'], '-f'],
                                 dict(os.environ, **self.mps_env), 'mps-controller')
        self.wait_controller(controller)
        self.mps_control(f'start_server -uid {os.getuid()}')
        self.wait_servers(controller)

    def off_domain(self):
        root = self.directory / 'mps-off'
        root.mkdir(mode=0o700)
        for name in ('pipe', 'log'):
            (root / name).mkdir(mode=0o700)
        self.mps_env = dict(CUDA_MPS_PIPE_DIRECTORY=str(root / 'pipe'),
                            CUDA_MPS_LOG_DIRECTORY=str(root / 'log'))
        self.persist()  # 私有空pipe避免OFF服务意外连接默认/他人的daemon。

    def wait_controller(self, controller):
        end = time.monotonic() + self.config['startup_seconds']
        while time.monotonic() < end:
            if controller.poll() is not None:
                raise SharedFault('MPS controller确定死亡')
            if (Path(self.mps_env['CUDA_MPS_PIPE_DIRECTORY']) / 'control').exists():
                return
            time.sleep(0.1)
        raise SharedFault('MPS controller未就绪')

    def wait_servers(self, controller):
        end = time.monotonic() + self.config['startup_seconds']
        while time.monotonic() < end:
            pids = self.mps_control('get_server_list').split()
            if pids and all(value.isdigit() for value in pids):
                records = [identity(int(value)) for value in pids]
                if any(not r or r['uid'] != os.getuid() or r['session'] != controller.pid for r in records):
                    raise SharedFault('MPS server归属不明')
                self.server_pids = [r['pid'] for r in records]
                self.persist()
                return
            time.sleep(0.1)
        raise SharedFault('未证实MPS server先行')

    def start_services(self):
        for model in self.config['models']:
            proc = self.launch(service_command(self.config, model), self.service_env(model), ipc.model_log_name(model['id']))
            self.services.append(proc)
            self.wait_ready(proc, model)
        self.probe('startup')

    def reserve_ipc(self, model):
        owned = ipc.allocation(model)
        name = owned['ipc_name']
        if ipc.paths(owned):
            raise SharedFault('私有IPC名称已存在，禁止复用')
        journal = ipc.journal_path(self.directory, name)
        journal.mkdir(mode=0o700)
        root = ipc.socket_root(name)
        root.mkdir(mode=0o700)
        owned['socket_root'] = ipc.evidence(root, root.lstat())
        self.allocations.append(owned)
        self.persist()
        write(journal / 'socket-root', owned['socket_root'])
        CreationJournal(journal, name).reserve_control()
        return name, journal

    def service_env(self, model):
        # UUID 在 MPS controller 和服务的不同 CUDA 序号空间中指向同一设备。
        mapping = self.config['logical_devices']
        devices = ','.join(mapping[str(index)] for index in model['placement'])
        name, journal = self.reserve_ipc(model)
        env = dict(os.environ)
        env.update(ENABLE_KVCACHED='1', KVCACHED_AUTOPATCH='1', PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   KVCACHED_IPC_NAME=name, KVREPRO_IPC_JOURNAL=str(journal), CUDA_VISIBLE_DEVICES=devices)
        for key in ('CUDA_MPS_PIPE_DIRECTORY', 'CUDA_MPS_LOG_DIRECTORY'):
            env.pop(key, None)
        env.update(self.mps_env)
        return env

    def wait_ready(self, proc, model):
        end = time.monotonic() + self.config['startup_seconds']
        while time.monotonic() < end:
            if proc.poll() is not None:
                raise RuntimeError('服务启动期间确定死亡')
            try:
                if healthy(model['port'], end) and time.monotonic() < end:
                    return
            except OSError:
                pass
            time.sleep(min(0.2, max(0, end - time.monotonic())))
        raise RuntimeError('服务启动就绪预算耗尽')

    def probe_environment(self):
        return dict(os.environ, KVREPRO_RESOURCE_DIRECTORY=str(self.directory),
                    KVREPRO_MPS_CONTROL=self.config['mps_control'], KVREPRO_NVIDIA_SMI=self.config['nvidia_smi'])

    def probe(self, phase):
        args = self.config['rank_probe']
        result = subprocess.run(args, env=self.probe_environment(),
              input=json.dumps(dict(models=self.config['models'],
              mps=self.row['mps'], server_pids=self.server_pids, resources=self.records)),
              text=True, capture_output=True, timeout=self.config['startup_seconds'])
        result.check_returncode()
        observed = json.loads(result.stdout)
        validate_ranks(self.config, self.row, observed, self.server_pids)
        sessions = {r['session'] for r in self.records}
        for rank in observed['ranks']:
            process = identity(rank['pid'])
            if process is None or process['session'] not in sessions:
                raise SharedFault('rank PID不属于自有服务session')
        write(self.directory / (phase + '-ranks.json'), observed)
        self.descendants = [member for root in self.records for member in session_members(root['session'])
                            if member['pid'] != root['pid']]
        self.persist()
        return observed

    def dead(self):
        return any(proc.poll() is not None for proc in self.processes if proc.pid != self.client_pid)

    def dead_services(self):
        if any(proc.poll() is not None for proc in self.processes
               if proc.pid != self.client_pid and proc not in self.services):
            return None  # 同时失去MPS等共享前提时，不能仅凭服务死亡判为局部。
        return [dict(pid=proc.pid, returncode=proc.returncode, observed_epoch=time.time())
                for proc in self.services if proc.poll() is not None]

    def client_code(self):
        return next((proc.poll() for proc in self.processes if proc.pid == self.client_pid), None)

    def sample(self):
        epoch, result = time.time(), None
        try:
            result = subprocess.run(self.config['kv_probe'], capture_output=True,
                                    env=self.probe_environment(),
                                    timeout=self.config['probe_seconds'], check=True)
            value = dict(epoch=epoch, observed=json.loads(result.stdout), error=None)
            value['derived'] = derive_kv(self.config, value['observed'])
            line = json.dumps(value, allow_nan=False)
        except (OSError, ValueError, subprocess.SubprocessError, LookupError, TypeError) as error:
            value = dict(epoch=epoch, observed=None, error=str(error), derived=None)
            source = error if result is None else result
            for stream in ('stdout', 'stderr'):
                data = getattr(source, stream, None)
                value[stream] = data.decode('utf-8', errors='surrogateescape') if isinstance(data, bytes) else data
            line = json.dumps(value, allow_nan=False)
        with (self.directory / 'kv.jsonl').open('a') as output:
            output.write(line + '\n')

    def close(self):
        failures = ['启动登记不完整'] if self.intent or len(self.processes) != len(self.records) else []
        diagnostics = self.close_processes(failures)
        self.close_handles()
        if not failures and self.allocations:
            try:
                cleanup_ipc(self.config, self.directory / 'resources.json')
            except Exception as error:
                failures.append('IPC清理不明: ' + str(error))
                diagnostics.append(cleanup_diagnostic(error, operation='cleanup_ipc'))
        evidence = dict(verified=not failures, errors=failures)
        if diagnostics:
            evidence['diagnostics'] = diagnostics
        write(self.directory / 'cleanup.json', evidence)
        if failures:
            raise SharedFault('; '.join(failures))

    def close_handles(self):
        for handle in self.handles:
            try:
                handle.close()
            except Exception as error:
                emit_cleanup_diagnostic(error, operation='close_log')
                raise

    def close_processes(self, failures):
        diagnostics = []
        for proc, record in reversed(list(zip(self.processes, self.records))):
            try:
                self.close_process(proc, record)
            except Exception as error:
                failures.append(str(error))
                diagnostics.append(cleanup_diagnostic(error, operation='close_process',
                                   pid=proc.pid, sid=record.get('session'), role=record.get('role')))
        return diagnostics

    def close_process(self, proc, record):
        with cleanup_operation('poll'):
            running = proc.poll() is None
        if running:
            with cleanup_operation('terminate'):
                terminate(record, self.config['cleanup_seconds'])
        else:
            with cleanup_operation('close_descendants'):
                self.close_descendants(record)
        with cleanup_operation('wait'):
            proc.wait(timeout=self.config['cleanup_seconds'])
        with cleanup_operation('verify_session_empty'):
            if session_members(record['session']):
                raise SharedFault('进程leader退出但session未空')

    def close_descendants(self, root):
        known = [r for r in self.descendants if r['session'] == root['session']]
        alive = []
        for record in known:
            with cleanup_operation('matches', pid=record['pid'], sid=record['session']):
                if matches(record):
                    alive.append(record)
        if alive:
            record = alive[0]
            with cleanup_operation('terminate', pid=record['pid'], sid=record['session']):
                terminate(record, self.config['cleanup_seconds'])
        if session_members(root['session']):
            raise SharedFault('leader死亡且剩余后代没有登记归属')


def service_command(config, model):
    args = [config['server_python'], '-s', str(Path(__file__).parents[1] / 'launch_server.py'),
            '--model-path', model['checkpoint'], '--host', '127.0.0.1', '--port', str(model['port']),
            '--base-gpu-id', '0', '--gpu-id-step', '1',
            '--tp-size', str(model['tp']), '--dp-size', str(model['dp']),
            '--context-length', str(model['context_length']),
            '--mem-fraction-static', str(model['mem_fraction_static']), '--disable-radix-cache']
    if model['dp_attention']:
        args.append('--enable-dp-attention')
    if model['max_total_tokens'] != 'auto':
        args += ['--max-total-tokens', str(model['max_total_tokens'])]
    return args + model.get('arguments', [])


def validate_ranks(config, row, observed, server_pids):
    expected = {(m['id'], rank) for m in config['models'] for rank in range(m['tp'])}
    ranks = observed['ranks']
    if len(ranks) != len(expected) or {(r['model'], r['rank']) for r in ranks} != expected:
        raise SharedFault('rank接入证据不完整')
    models = {m['id']: m for m in config['models']}
    for rank in ranks:
        model = models[rank['model']]
        if rank['gpu'] != config['logical_devices'][str(model['placement'][rank['rank']])]:
            raise SharedFault('rank设备映射不符')
        if rank['mps_server_pid'] not in (server_pids if row['mps'] else [None]):
            raise SharedFault('MPS接入状态不符')
        if rank['prefix_cache_enabled'] is not False or rank['dp_attention'] != model['dp_attention']:
            raise SharedFault('缓存/DP attention实际状态不符')
        if rank.get('kvcached_enabled') is not True:
            raise SharedFault('缺少实际kvcached启用证据')
        validate_rank_geometry(model, rank)


def validate_rank_geometry(model, rank):
    expected = model['geometry_by_rank'][str(rank['rank'])]
    if rank.get('geometry') != expected:
        raise SharedFault('实际rank KV几何未证实或与配置不符')
    capacity = rank.get('capacity_tokens')
    if type(capacity) is not int or capacity <= 0:
        raise SharedFault('实际服务容量未知，不能假定已配置虚拟上界均可使用')
    if model['max_total_tokens'] != 'auto' and capacity > model['max_total_tokens']:
        raise SharedFault('实际容量超过显式虚拟上界')


def derive_kv(config, observed):
    models = {m['id']: m for m in config['models']}
    values = []
    for row in observed['ranks']:
        model = models[row['model']]
        if row['gpu'] != config['logical_devices'][str(model['placement'][row['rank']])]:
            raise ValueError('KV采样GPU归属与冻结配置不符')
        geometry = model['geometry_by_rank'][str(row['rank'])]
        logical = kv_bytes(row['active_tokens'], geometry) if row.get('active_tokens') is not None else None
        values.append(dict(model=row['model'], rank=row['rank'], gpu=row['gpu'],
                           logical_bytes=logical, physical_bytes=row.get('physical_bytes'),
                           observed_epoch=row.get('observed_epoch'),
                           geometry=geometry, basis='配置几何推断；物理值为原始探针观察'))
    return values


def cleanup_ipc(config, registry):
    entrypoint = Path(__file__).resolve().parents[1] / 'main.py'
    args = [config['server_python'], '-s', str(entrypoint), 'batch-ipc-clean', '--registry', str(registry)]
    env = dict(os.environ, ENABLE_KVCACHED='0', KVCACHED_AUTOPATCH='0', KVCACHED_IPC_NAME='kvrepro_cleanup')
    result = subprocess.run(args, env=env, text=True, capture_output=True, timeout=config['cleanup_seconds'])
    write(registry.parent / ('ipc-cleanup-' + str(time.time_ns()) + '.json'),
          dict(command=args, code=result.returncode, stdout=result.stdout, stderr=result.stderr))
    result.check_returncode()
