import time
import taskgen.task as task

class Job(task.Job):
    """Extended Job class that allows to simulate execution on a proxy scheduler"""
    def __init__(self, parenttask):
        super().__init__()
        self._remaining_executiontime = parenttask['executiontime']
        self.slice = 0
        self.parenttask = parenttask
        self.id = parenttask.jobcount

    def run(self, timestamp, quantum):
        if self.start_date is None:
            self.start_date = timestamp

        self._remaining_executiontime -= quantum
        self.slice += quantum

        if self._remaining_executiontime == 0:
            self.end_date = timestamp + quantum


class Task(task.Task):
    """Extended Task class that allows to simulate execution on a proxy scheduler"""

    def __init__(self, *blocks):
        self.lifetime = 0
        self['offset'] = 0 #release time
        super().__init__(*blocks)

    def run(self, quantum):
        self.lifetime += quantum

    def has_job(self):
        return self['numberofjobs'] > self.job_count
    
    def has_ready_job(self):
        return self.has_job() and self.lifetime 

    def next_job(self):
        job = Job(self)
        self.jobs.append(job)
        return job
    
    def is_finished(self):
        return self.job_count >= self['numberofjobs'] and not self.jobs[-1].is_running()
    
    @property
    def job_count(self):
        return len(self.jobs)