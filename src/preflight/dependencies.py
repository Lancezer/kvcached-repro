"""发行包信息及基础直接依赖核对。"""
from importlib import metadata


def packages():
    result = {}
    for name in ('kvcached', 'sglang', 'torch', 'transformers', 'triton',
                 'flashinfer-python', 'wrapt'):
        dist = metadata.distribution(name)
        result[name] = {'version': dist.version, 'root': str(dist.locate_file(''))}
    return result


def requirement_issue(name, text):
    from packaging.requirements import Requirement
    req = Requirement(text)
    if req.marker and not req.marker.evaluate({'extra': ''}):
        return None
    try:
        installed = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        installed = None
    if installed is None or installed not in req.specifier:
        return {'package': name, 'requirement': text, 'installed': installed}
    return None


def dependency_issues():
    # 仅核对基础直接依赖，不扩展到 extras、传递依赖或 ABI。
    issues = []
    for name in ('kvcached', 'sglang', 'torch', 'transformers'):
        for text in metadata.requires(name) or []:
            issue = requirement_issue(name, text)
            if issue:
                issues.append(issue)
    return issues
