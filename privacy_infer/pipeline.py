from .errors import PrivacyError


class Pipeline:
    def __init__(self, gate, cloud):
        self.gate, self.cloud = gate, cloud

    def run(self, task, context, *, dry_run=True):
        request, report = self.gate.prepare(task, context)
        if dry_run:
            return request, report, None
        # There is intentionally no fallback that sends the original inputs.
        return request, report, self.cloud.complete(request)
