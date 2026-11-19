"""按配对预算限制批次中新 case 的准入时间。"""
import time


class Window:
    def __init__(self, end, pair_budget, cleanup_budget, clock=time.time):
        if not end > clock() or not pair_budget > cleanup_budget > 0:
            raise ValueError('invalid reservation window')
        self.stop = end - cleanup_budget
        self.pair_budget = pair_budget
        self.clock = clock
        self.active_pair = None

    def admit(self, row, index):
        pair = row['pair_id']
        if index['cases'].get(row['case_id'], {}).get('adopted'):
            return
        now = self.clock()
        if now >= self.stop or (pair != self.active_pair and now + self.pair_budget > self.stop):
            raise InterruptedError('insufficient reservation time for pending pair')
        self.active_pair = pair

    def install(self, runner):
        original = runner.drive_case

        def drive_case(config, row, output, index, selected, reason):
            self.admit(row, index)
            return original(config, row, output, index, selected, reason)

        runner.drive_case = drive_case
