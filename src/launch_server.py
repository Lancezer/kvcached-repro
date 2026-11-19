"""批次服务启动入口；spawn子解释器也先补齐经过内核核验的OFD常量。"""
import runpy
from ofd_compat import enable
from batch.ipc_journal import install
from kv_export import install as install_kv_export

enable()
install()
install_kv_export()

if __name__ == '__main__':
    runpy.run_module('sglang.launch_server', run_name='__main__')
