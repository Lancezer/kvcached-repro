import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import kv_probe


class KVProbeClockTests(unittest.TestCase):
    def test_each_rank_uses_clock_after_atomic_snapshot_update(self):
        models = [dict(id='model', checkpoint='checkpoint', tp=2)]
        resources = dict(records=[dict(session=42)])
        epoch = 100.0

        def snapshot(rank):
            pid = rank + 10
            return dict(pid=pid, rank=rank, checkpoint='checkpoint',
                        process_identity=dict(pid=pid), observed_epoch=epoch)

        original_read = Path.read_text

        def publish_and_read(path):
            nonlocal epoch
            epoch += 6
            replacement = path.with_suffix('.tmp')
            replacement.write_text(json.dumps(snapshot(int(path.stem) - 10)))
            replacement.replace(path)
            epoch += 0.25
            return original_read(path)

        with tempfile.TemporaryDirectory() as directory:
            for rank in range(2):
                (Path(directory) / f'{rank + 10}.json').write_text(json.dumps(snapshot(rank)))
            with (patch.object(Path, 'read_text', publish_and_read),
                  patch('kv_probe.time.time', side_effect=lambda: epoch) as clock,
                  patch('kv_probe.matches', return_value=True),
                  patch('kv_probe.identity', return_value=dict(session=42))):
                rows = kv_probe.rows(dict(models=models), directory, resources)
        self.assertEqual({row['rank'] for row in rows}, {0, 1})
        self.assertEqual(clock.call_count, 2)

    def test_freshness_interval_includes_both_endpoints(self):
        for observed in (100.0, 95.0):
            with self.subTest(observed=observed):
                kv_probe.validate_freshness('10.json', dict(pid=10, observed_epoch=observed), 100.0)

    def test_rejected_observations_report_distinct_reasons_and_timing(self):
        cases = [
            (100.25, None, 'negative age', '-0.25'),
            (94.75, None, 'stale observation', '5.25'),
            (100.25, 'capture failed', "exporter reported error 'capture failed'", '-0.25'),
            (float('nan'), None, 'invalid observation timestamp', 'nan'),
        ]
        for observed, error, reason, age in cases:
            with self.subTest(reason=reason):
                value = dict(pid=10, observed_epoch=observed, error=error)
                with self.assertRaises(ValueError) as caught:
                    kv_probe.validate_freshness('10.json', value, 100.0)
                for field in (reason, 'pid=10', "source='10.json'", f'observed_epoch={observed!r}',
                              'checked_epoch=100.0', f'age_seconds={age}'):
                    self.assertIn(field, str(caught.exception))
