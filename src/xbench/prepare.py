"""仅修改独立 CrossPool 副本的两处延迟导入。"""


def delayed_import(text, old, target, indentation):
    if text.count(old) != 1 or text.count(target) != 1 or '\n' + old not in text:
        raise ValueError('需要固定版本的原始 CrossPool 源码')
    return text.replace(old, '').replace(target, indentation + old + '\n' + target)


def prepare(source, catalog):
    config = source / 'src/xpool/config.py'
    suite = source / 'benches/suites/serving/multi_model.py'
    patched = delayed_import(config.read_text(), 'import xpool.native\n',
                             '        return xpool.native.debug.Options(\n', '        ')
    text = delayed_import(suite.read_text(),
                          'from xkit.serving.sglang.system import XpoolServingSystem\n',
                          '                    system = XpoolServingSystem()', '                    ')
    destination = catalog / 'suites/serving/multi_model.py'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    config.write_text(patched)
