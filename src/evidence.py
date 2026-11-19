"""保存 JSON 原始证据。"""
import json


def save(out, name, data):
    (out / name).write_text(json.dumps(data, indent=2, ensure_ascii=False))
