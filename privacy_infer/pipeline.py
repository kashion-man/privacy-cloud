class Pipeline:
    def __init__(self, gate, cloud):
        self.gate, self.cloud = gate, cloud

    def run(self, task, context, *, dry_run=True, approve=None):
        request, report = self.gate.prepare(task, context)
        if dry_run:
            return request, report, None
        return request, report, self.cloud.complete(request, approve=approve)
