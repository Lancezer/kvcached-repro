"""会话恢复、最近两轮模板和会话等权抽轮；不生成随机到达计划。"""
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from .source import verify
from localpaths import allowed


def stable(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def messages(row):
    result = []
    for value in row['conversations']:
        if not isinstance(value, dict):
            result.append(('invalid', ''))
            break
        role, text = value.get('from'), value.get('value')
        if not isinstance(role, str) or not isinstance(text, str):
            result.append(('invalid', ''))
            break
        result.append((role, text))
    return result


def group_fragments(rows, pattern):
    groups, rejected, poisoned = defaultdict(list), Counter(), set()
    expression = re.compile(pattern)
    if set(expression.groupindex) != {'conversation', 'part'}:
        raise ValueError('ID规则须显式给出conversation和part两个命名组')
    for row in rows:
        match = expression.fullmatch(str(row.get('id', '')))
        if not match:
            rejected['unmapped_id'] += 1
            continue
        try:
            groups[match['conversation']].append((int(match['part']), messages(row), str(row['id'])))
        except (ValueError, KeyError, TypeError):
            rejected['invalid_fragment'] += 1
            poisoned.add(match['conversation'])
    for conversation in poisoned:
        groups.pop(conversation, None)
    return groups, rejected


def join_fragments(fragments, first_part, mapping=None):
    unique = {}
    for part, values, *_ in fragments:
        if part in unique and unique[part] != values:
            raise ValueError('同偏移片段冲突')
        unique[part] = values
    parts = sorted(unique)
    if not parts or parts[0] != first_part or first_part != 0:
        raise ValueError('缺少起点')
    merged = []
    for part in parts:
        merged = append_at_offset(merged, unique[part], part)
    if mapping is not None:
        mapping.extend(dict(fragment_reference(f), merged_offset=f[0],
                            basis='publisher_message_offset') for f in fragments)
    return merged


def append_at_offset(previous, following, offset):
    # 发布方后缀是消息绝对偏移；无重叠邻接合法，重复正文也无需猜测对齐。
    if offset > len(previous):
        raise ValueError('消息偏移缺口')
    overlap = min(len(previous) - offset, len(following))
    if previous[offset:offset + overlap] != following[:overlap]:
        raise ValueError('消息偏移内容冲突')
    return previous + following[overlap:]


def fragment_reference(fragment):
    return dict(part=fragment[0], fragment_id=fragment[2] if len(fragment) > 2 else str(fragment[0]),
                source_message_positions=list(range(len(fragment[1]))), messages_sha256=stable(fragment[1]))


def valid_pairs(values):
    pairs = []
    for offset in range(0, len(values) - 1, 2):
        question, answer = values[offset:offset + 2]
        if question[0] != 'human' or answer[0] != 'gpt':
            break
        if not question[1].strip() or not answer[1].strip():
            break
        pairs.append((question[1], answer[1]))
    return pairs


def prompt(pairs, turn):
    lines = []
    for question, answer in pairs[max(0, turn - 2):turn]:
        lines.extend(['Human: ' + question, 'Assistant: ' + answer])
    lines.extend(['Human: ' + pairs[turn][0], 'Assistant:'])
    return '\n'.join(lines)


def candidates(groups, first_part, exclusions=None):
    result, rejected = {}, Counter()
    for conversation in sorted(groups):
        try:
            mapping = []
            restored = join_fragments(groups[conversation], first_part, mapping)
            pairs = valid_pairs(restored)
            discarded = len(restored) - 2 * len(pairs)
            rejected['discarded_suffix_messages'] += discarded
            if discarded:
                exclude(exclusions, conversation, groups[conversation], 'discarded_suffix_messages', discarded_messages=discarded, retained_pairs=len(pairs))
        except ValueError as error:
            rejected[str(error)] += 1
            exclude(exclusions, conversation, groups[conversation], str(error))
            continue
        if not pairs:
            rejected['no_complete_pair'] += 1
            exclude(exclusions, conversation, groups[conversation], 'no_complete_pair')
            continue
        result[conversation] = turn_candidates(conversation, pairs, mapping)
    return result, rejected


def exclude(exclusions, conversation, fragments, reason, **details):
    if exclusions is not None:
        exclusions.append(dict(conversation=conversation, reason=reason, **details,
                               fragments=[fragment_reference(f) for f in fragments]))


def turn_candidates(conversation, pairs, mapping):
    return [dict(conversation=conversation, turn=i, text=prompt(pairs, i),
                 fragments=mapping, source_message_positions=[2 * i, 2 * i + 1],
                 history_turns=min(2, i), reference_sha256=stable(pairs[i][1]))
            for i in range(len(pairs))]


def load_tokenizers(models):
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer
    result = {}
    for model in models:
        if not Path(model['tokenizer']).is_dir():
            raise ValueError('tokenizer必须为已有本地目录')
        result[model['id']] = get_tokenizer(model['tokenizer'], local_files_only=True, trust_remote_code=True)
    return result


def eligible(candidate, tokenizers, models, output_tokens):
    counts = {}
    for model in models:
        tokenizer = tokenizers[model['id']]
        # 与服务的特殊token行为须在配置显式给定，并由后续真实短验核对。
        count = len(tokenizer.encode(candidate['text']))
        if count >= model['input_max_exclusive'] or count + output_tokens > min(model['context_length'], model['request_total_limit']):
            return None
        counts[model['id']] = count
    return dict(candidate, tokens=counts)


def select(pool, tokenizers, config):
    selected, rejected = [], Counter()
    rng = random.Random(config['data_seed'])
    for conversation in sorted(pool):
        choices = [eligible(c, tokenizers, config['models'], config['output_tokens']) for c in pool[conversation]]
        valid = [c for c in choices if c is not None]
        rejected['length_filtered_turns'] += len(choices) - len(valid)
        if not valid:
            rejected['no_intersection_conversation'] += 1
            continue
        candidate = dict(rng.choice(valid))
        candidate['prompt_id'] = stable([conversation, candidate['turn'], candidate['text']])
        selected.append(candidate)
    return selected, rejected


def lengths(values):
    ordered = sorted(values)
    if not ordered:
        return None
    def quantile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return dict(min=ordered[0], max=ordered[-1], **{f'P{q}': quantile(q / 100) for q in (50, 90, 95, 99)})


def write_pool(destination, selected, config, provenance, counts, exclusions=()):
    destination.mkdir(exist_ok=False)
    if not selected:
        raise ValueError('三tokenizer交集为空，不能生成空输入池')
    write_jsonl(destination / 'prompts.jsonl', [dict(prompt_id=c['prompt_id'], text=c['text']) for c in selected])
    write_jsonl(destination / 'mapping.jsonl', [{k: v for k, v in c.items() if k != 'text'} for c in selected])
    write_jsonl(destination / 'excluded.jsonl', exclusions)
    from .source import checksum
    files = {name: checksum(destination / name) for name in ('prompts.jsonl', 'mapping.jsonl', 'excluded.jsonl')}
    from batch.config import verify_inventory
    tokenizers = {m['id']: verify_inventory(dict(m, checkpoint=m['tokenizer'])) for m in config['models']}
    manifest = dict(source=provenance, rules=config, counts=counts, size=len(selected), files=files, tokenizer_assets=tokenizers,
                    processor=processor_identity(),
                    history=dict(Counter(c['history_turns'] for c in selected)),
                    history_fraction={str(i): sum(c['history_turns'] == i for c in selected) / len(selected) for i in range(3)},
                    lengths={m['id']: lengths([c['tokens'][m['id']] for c in selected]) for m in config['models']})
    with (destination / 'pool.json').open('x') as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False, allow_nan=False)
    return manifest


def processor_identity():
    import importlib.metadata
    import sys
    from .source import checksum
    paths = (Path(__file__), Path(__file__).with_name('source.py'))
    return dict(code={p.name: checksum(p) for p in paths}, python=sys.version,
                packages={name: importlib.metadata.version(name) for name in ('sglang', 'transformers')})


def write_jsonl(path, rows):
    with path.open('x') as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def prepare(source, destination, config):
    validate_config(config)
    provenance = verify(source)
    rows = json.loads(allowed(source).read_text())
    groups, rejected = group_fragments(rows, config['fragment_id_pattern'])
    exclusions = []
    pool, invalid = candidates(groups, config['first_part'], exclusions)
    selected, filtered = select(pool, load_tokenizers(config['models']), config)
    counts = dict(rows=len(rows), grouped_fragments=sum(map(len, groups.values())),
                  grouped_conversations=len(groups), prefix_conversations=len(pool),
                  prefix_turns=sum(map(len, pool.values())), selected=len(selected),
                  rejected=dict(rejected + invalid + filtered))
    return write_pool(allowed(destination), selected, config, provenance, counts, exclusions)


def validate_config(config):
    if not isinstance(config.get('fragment_id_pattern'), str) or type(config.get('first_part')) is not int:
        raise ValueError('必须按固定文件实际ID规则显式填写片段映射与起点，不能猜测')
    if type(config['data_seed']) is not int or type(config['output_tokens']) is not int or config['output_tokens'] <= 0:
        raise ValueError('data_seed/输出预算非法')
    ids = [model['id'] for model in config['models']]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('tokenizer模型ID为空或重复')
    for model in config['models']:
        allowed(model['tokenizer'])
        if model.get('tokenizer_semantics') != 'sglang-default-encode':
            raise ValueError('输入计数必须显式确认与固定原生SGLang encode语义一致')
        if not isinstance(model.get('assets_manifest'), str):
            raise ValueError('必须提供完整tokenizer/checkpoint文件SHA256清单')
        if type(model.get('input_max_exclusive')) is not int or model['input_max_exclusive'] <= 0:
            raise ValueError('必须提供实际服务的输入独占上限')
        if type(model.get('request_total_limit')) is not int or model['request_total_limit'] <= config['output_tokens']:
            raise ValueError('必须提供实际服务可接受的输入+输出边界')
