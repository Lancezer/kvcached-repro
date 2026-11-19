"""单一 CLI；直接执行源码或独立 zipapp。"""
import argparse
from pathlib import Path
from runtime import options, configuration


def parser():
    cli = argparse.ArgumentParser(description='kvcached 复现工具；运行 GPU 前须重新协调设备')
    commands = cli.add_subparsers(dest='action', required=True)
    from batch.cli import configure
    configure(commands)
    commands.add_parser('preflight', help='目标环境的隔离禁用预检')
    validate = commands.add_parser('validate', help='目标主机服务 OFF/ON 验证')
    validate.add_argument('--out', required=True, type=Path)
    options(validate)
    prepare = commands.add_parser('prepare', help='准备独立 xbench 副本')
    prepare.add_argument('source', type=Path)
    prepare.add_argument('catalog', type=Path)
    smoke = commands.add_parser('smoke', help='目标主机 xbench 单请求')
    smoke.add_argument('--root', required=True, type=Path)
    smoke.add_argument('--out', required=True, type=Path)
    options(smoke)
    return cli


def main():
    args = parser().parse_args()
    if args.action.startswith(('batch-', 'data-')) or args.action == 'prepare-frozen':
        from batch.cli import dispatch
        return dispatch(args)
    if args.action == 'preflight':
        from preflight.check import main as preflight
        return preflight()
    if args.action == 'prepare':
        from xbench.prepare import prepare
        prepare(args.source.resolve(), args.catalog.resolve())
        return 0
    if args.action == 'validate':
        from service.validation import main as validate
        return validate(args.out.resolve(), configuration(args))
    from xbench.smoke import main as smoke
    return smoke(args.root, args.out, configuration(args))


def entrypoint():
    # zipapp 默认忽略返回值；统一在这里传递 CLI 退出码。
    raise SystemExit(main())


if __name__ == '__main__':
    entrypoint()
