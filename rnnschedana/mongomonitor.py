import loggingconfig
from pymongo import MongoClient
from mongointerface import MongoInterface
from taskgen.monitor import AbstractMonitor
from utilities import truncate


#For the moment I'm ignoring that the callbacks are called from multiple threads, mongodb should be able to handle this
class MongoMonitor(AbstractMonitor):
    """Stores task-sets and events to a MongoDB"""
    def __init__(self, sample_coll):
        #create _logger at instance level, because it doesn't show at module level for some reason (?)
        self._logger = loggingconfig.getLogger('MongoMonitor')
        self._sample_coll = sample_coll

    def taskset_event(self, taskset):
        self._logger.debug('Taskset_event: ' + truncate(taskset.__str__()))

    def taskset_start(self, taskset):
        self._logger.debug('Taskset_start: ' + truncate(taskset.__str__()))

    def taskset_finish(self, taskset):
        self._logger.debug('Taskset_finish: ' + truncate(taskset.__str__()))

    def taskset_stop(self, taskset):
        self._logger.debug('Taskset_stop: ' + truncate(taskset.__str__()))

    def task_event(self, task):
        self._logger.debug('task_event: ' + truncate(task.__str__()))

    def task_stop(self, task):
        self._logger.debug('task_stop: ' + truncate(task.__str__()))

    def task_start(self, task):
        self._logger.debug('task_start: ' + truncate(task.__str__()))

    def task_finish(self, task):
        self._logger.debug('task_finish: ' + truncate(task.__str__()))
        self._sample_coll.insert_one(task)
