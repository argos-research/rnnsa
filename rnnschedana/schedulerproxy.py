from queue import PriorityQueue
from queue import Queue
from queue import Empty
import pprint
from mongomonitor import MongoMonitor
import loggingconfig
from collections import Iterable
from taskgen.task import Task
from taskgen.task import Job
from taskgen.taskset import TaskSet

def queueiter(queue):
    while True:
        try:
            yield queue.get_nowait()
        except:
            return

def put_coll(queue, coll):
    if isinstance(coll, Queue):
        coll = queueiter(coll)
    if isinstance(coll, Iterable):
        for item in coll:
           queue.put(item)
    else:
        queue.put(coll)

class ReadyQueue(PriorityQueue):
    """PriorityQueue specifically designed for (priority, Queue<JCB>) tuples while allowing only one tuple per priority.
    If a tuple exists for any given priority, this priorities' queue is simply extended"""

    def _init(self, maxsize):
        """overwrite _init and add priority:value dict to keep track of items"""
        PriorityQueue._init(self, maxsize)
        self._prio_levels = {}

    def _put(self, item):
        """ReadyQueue expects (priority, Queue<JCB>), (priority, list<JCB>) or (priority, JCB) tuples"""
        p, q = item

        if p in self._prio_levels.keys():
            prio_level_queue = self._prio_levels[p]
            put_coll(prio_level_queue, q)

        else:
            prio_level_queue = Queue()
            put_coll(prio_level_queue, q)
            PriorityQueue._put(self, (p, prio_level_queue))
            self._prio_levels[p] = prio_level_queue

    def _get(self):
        """Remove queue from both PQ and items dictionary"""
        p, q = PriorityQueue._get(self)
        del self._prio_levels[p]
        return p, q

class TCB:
    def __init__(self, task):
        #a TCB encapsulates a task
        self.task = task

        #every TCB is parent to a number of 'child' JCBs
        self.jcbs = []

        #bookkeeping variables
        self._lifetime = 0
        self._last_running_ts = None
        self._interrupted = True

        #everything we want to save in the database
        self.task['activation_ts'] = []
        self.task['interrupt_ts'] = []
        self.task['start_date'] = None
        self.task['end_date'] = None
        self.task['success'] = True

        #logging
        self._logger = loggingconfig.getLogger('SCH.TCB_{}'.format(self.task['id']))

    def live(self, quantum):
        self._lifetime += quantum

    def run(self, timestamp, quantum):
        self._last_running_ts = timestamp + quantum

    def has_job(self):
        return self.task['numberofjobs'] > self.job_count

    def has_ready_job(self):
        if self.task['period']:
            return self.has_job() and self._lifetime > self.job_count * self.task['period']
        else:
            return self.has_job()

    def next_job(self):
        jcb = JCB(self)
        self.jcbs.append(jcb)
        return jcb

    @property
    def youngest_jcb(self):
        return self.jcbs[-1]

    def log_success(self, bool_value):
        self.task['success'] = bool_value

    def log_end_date(self, timestamp):
        self.task['end_date'] = timestamp

    def log_start_date(self, timestamp):
        self.task['start_date'] = timestamp

    def log_activation(self, timestamp):
        self.task['activation_ts'].append(timestamp)

    def log_interrupt(self):
        self.task['interrupt_ts'].append(self._last_running_ts)

    def log_critical_end_date(self):
        self.task['end_date'] = self.task['start_date'] + self.task['criticaltime']

    def had_interrupt(self, timestamp):
        if self._last_running_ts == None:
            return False
        return self._last_running_ts < timestamp

    @property
    def has_started(self):
        return not self.task['start_date'] == None

    @property
    def has_finished(self):
        return self.job_count >= self.task['numberofjobs'] and self.jcbs[-1].has_finished

    @property
    def is_running(self):
        #return len(self.task['activation_ts']) > len(self.task['interrupt_ts'])
        return not self._interrupted

    @property
    def is_interrupted(self):
        return self._interrupted

    @property
    def past_critical(self):
        return self._lifetime >= self.task['criticaltime']

    @property
    def job_count(self):
        return len(self.jcbs)

    def __str__(self):
        return 'TCB_{} [prio: {}, jobs: {}/{}, lt: {}/{}]'.format(
            self.task['id'],
            self.task['priority'],
            self.job_count,
            self.task['numberofjobs'],
            self._lifetime,
            self.task['criticaltime'])

class JCB:
    def __init__(self, parentTCB):
        #link to parent task
        self.parentTCB = parentTCB

        #bookkeeping variables
        self._remaining_executiontime = self.parentTCB.task['executiontime']
        self._slice = 0
        self._last_running_ts = None
        self._deadline = None
        self._interrupted = True

        #everything we want to save in the database
        self.job = Job()
        self.job['activation_ts'] = []
        self.job['interrupt_ts'] = []
        self.job['id'] = self.parentTCB.job_count
        self.job['success'] = False
        self.job['release'] = None
        self.parentTCB.task['jobs'].append(self.job)

        #logging
        self._logger = loggingconfig.getLogger('SCH.TCB_{}.JCB_{}'.format(self.parentTCB.task['id'], self.job['id']))

    def run(self, timestamp, quantum):
        #update bookkeeping variables
        self._remaining_executiontime -= quantum
        self._slice += quantum
        self._last_running_ts = timestamp + quantum
    
    def log_release(self, timestamp):
        self.job['release'] = timestamp
        self._deadline = timestamp + self.parentTCB.task['criticaltime']

    def log_success(self, bool_value):
        self.job['success'] = bool_value

    def log_end_date(self, timestamp):
        self.job['end_date'] = timestamp

    def log_critical_end_date(self):
        """CALL ONLY AFTER PARENT log_critical_end_date HAS BEEN CALLED"""
        self.job['end_date'] = self.parentTCB.task['end_date']

    def log_start_date(self, timestamp):
        self.job['start_date'] = timestamp

    def log_activation(self, timestamp):
        self.job['activation_ts'].append(timestamp)
        self._interrupted = False

    def log_interrupt(self):
        self.job['interrupt_ts'].append(self._last_running_ts)
        self._interrupted = True

    def due_timer_interrupt(self, timer):
        if self._slice >= timer:
            self._slice = 0
            return True
        return False

    def had_interrupt(self, timestamp):
        if self._last_running_ts == None:
            return False
        return self._last_running_ts < timestamp

    @property
    def has_started(self):
        return not self.job['start_date'] == None

    @property
    def has_finished(self):
        return self._remaining_executiontime <= 0

    @property
    def is_running(self):
        return not self._interrupted
        #return len(self.job['activation_ts']) > len(self.job['interrupt_ts'])

    @property
    def is_youngest(self):
        return self.job['id'] == self.parentTCB.job_count - 1

    @property
    def is_interrupted(self):
        return self._interrupted

    def past_critical(self, timestamp):
        return timestamp > self._deadline

    def __str__(self):
        return 'TCB_{}.JCB_{} [prio: {}, start: {}, rem: {}]'.format(
            self.parentTCB.task['id'],
            self.job['id'],
            self.parentTCB.task['priority'],
            self.job['start_date'],
            self._remaining_executiontime)

class PriorityRoundRobin2:
    def __init__(self, quantum, timer, monitor):
        self._init(quantum, timer, monitor) #I thought I needed this in another place...

    def _init(self, quantum, timer, monitor):
        #configuration variables
        self._quantum = quantum #minimum atomic runtime of a job
        self._timer = timer #max time slice per job until timer interrupt
        self._timestamp = 0 #start time

        #bookkeeping
        self._ready_jobs_queue = ReadyQueue() #ReadyQueue of (priority, Queue<JCB>) tuples
        self._ready_tasks_set = set() #List of currently running tasks (i.e. tasks with uncompleted jobs) -> should be an ordered set (?)

        #saving
        self._monitor = monitor #monitor to log results

        #logging
        self._logger = loggingconfig.getLogger('SCH')

    def _run(self):
        """from current state, run simulation for one time quantum"""
        #fetch possibly ready new jobs from currently running tasks and append to queue
        for tcb in self._ready_tasks_set:
            while tcb.has_ready_job():
                next_jcb = tcb.next_job()
                next_jcb.log_release(self._timestamp)
                self._ready_jobs_queue.put((tcb.task['priority'], next_jcb))
                self._logger.debug('ts {}: enqueued {}'.format(self._timestamp, next_jcb))

        try:
            #get highest priority job queue
            p, q = self._ready_jobs_queue.get_nowait()

            #get current job
            jcb = q.queue[0]

            if jcb.past_critical(self._timestamp):
                self._logger.info('ts {}: {} is past critical. Abort simulation.'.format(self._timestamp, jcb))
                jcb.log_success(False)
                jcb.parentTCB.log_end_date(self._timestamp + self._quantum)
                return False
                #ABORT SIMULATION. JOB WAS NOT ABLE TO COMPLETE IN TIME -> TASKSET NOT SCHEDULABLE!

            self._logger.debug('ts {}: Scheduling {}'.format(self._timestamp, jcb))

            #update state
            #if run for the first time, log start_date
            if not jcb.has_started:
                jcb.log_start_date(self._timestamp)

            #check for priority interrupt
            if jcb.is_running and jcb.had_interrupt(self._timestamp):
                jcb.log_interrupt()
                jcb.parentTCB.log_interrupt()
            #if not resuming execution in timeslice log activation
            if jcb.is_interrupted:
                jcb.log_activation(self._timestamp)
                jcb.parentTCB.log_activation(self._timestamp)

            #update state
            jcb.run(self._timestamp, self._quantum)
            jcb.parentTCB.run(self._timestamp, self._quantum)

            #case: this job is finished
            if jcb.has_finished:
                jcb.log_end_date(self._timestamp + self._quantum)
                jcb.log_interrupt()
                jcb.log_success(True)
                try:
                    q.get_nowait()
                except Empty:
                    self._logger.error('ts {}: This should never happen'.format(self._timestamp))

                self._logger.info('ts {}: {} is finished.'.format(self._timestamp, jcb))

                jcb.parentTCB.log_interrupt()
                #case: this job is finished AND it was the last job of this task
                if jcb.parentTCB.has_finished:
                    jcb.parentTCB.log_end_date(self._timestamp + self._quantum)
                    jcb.parentTCB.log_success(True)
                    self._ready_tasks_set.remove(jcb.parentTCB)
                    self._logger.info('ts {}: This was the last job -> {} is finished.'.format(self._timestamp, jcb.parentTCB))
                    self._logger.info('ts {}: {} saved.'.format(self._timestamp, jcb.parentTCB))

            #case: this job is not finished yet
            else:
                #check for due timer interrupt
                if jcb.due_timer_interrupt(self._timer):
                    #move to end of this priorities' queue
                    try:
                        q.get_nowait()
                        q.put(jcb)
                    except Empty:
                        self._logger.error('ts {}: This should never happen'.format(self._timestamp))

                    #log timer interrupt
                    jcb.log_interrupt()
                    jcb.parentTCB.log_interrupt()

                    self._logger.debug('ts {}: {} timer interrupt -> RR re-queue'.format(self._timestamp, jcb))

            #if this prio level's queue is not empty put tuple back into the readyqueue
            if not q.empty():
                self._ready_jobs_queue.put((p, q))

        except Empty:
            #if there are no jobs to be scheduled, just advance doing nothing
            #self._logger.info('ts {}: ReadyQueue is empty'.format(self._timestamp))
            pass

        #update all currently running tasks on the machine
        remove_set = []
        for tcb in self._ready_tasks_set:
            if not tcb.has_started:
                tcb.log_start_date(self._timestamp)
            tcb.live(self._quantum)

        #advance simulation one quantum
        self._timestamp += self._quantum
        return True

    def schedule(self, tcb):
        """expects a task with an release higher than all previous tasks,
        and advances the simulation until this task's release, s.th. it can be enqueued"""

        if tcb.task['release'] <= self._timestamp - self._quantum:
            raise ValueError('task must not start earlier than the current simulation timestamp')

        while self._timestamp < tcb.task['release']:
            if not self._run():
                return False

        self._logger.info('ts {}: Scheduling task {}...'.format(self._timestamp, tcb.task['id']))
        self._ready_tasks_set.add(tcb)
        self._logger.debug('ts {}: {} tasks in ReadyQueue'.format(self._timestamp, len(self._ready_tasks_set)))
        return True

    def finish(self):
        """runs the simulation until all current tasks are finished"""
        self._logger.info('ts {}: Finishing tasks...'.format(self._timestamp))
        while self._ready_tasks_set and not self._ready_jobs_queue.empty():
            if not self._run():
                return False
        return True

    def _save(self, tcbs):
        for tcb in tcbs:
            self._monitor.task_finish(tcb.task)

    def run_sample(self, sample, size):
        """simulate scheduling of a sample set of tasks"""
        if not isinstance(sample, Iterable):
            raise TypeError('sample must be iterable')

        if not sample:
            raise Exception('sample must not be empty')

        self._logger.info('ts {}: Sample Simulation Start.'.format(self._timestamp))
        tcbs = []
        tcb = None
        for i in range(size):
            tcb = TCB(next(sample))
            tcbs.append(tcb)
            if not self.schedule(tcb):
                tcb.log_success(False)
                self._logger.info('ts {}: Sample Simulation Done. Sample not schedulable after {}'.format(self._timestamp, tcb))
                self._save(tcbs)
                return False

        if not self.finish():
            self._logger.info('ts {}: Sample Simulation Done. Sample not schedulable after {}'.format(self._timestamp, tcb))
            self._save(tcbs)
            return False

        self._logger.info('ts {}: Sample Simulation Done. Sample completely schedulable.'.format(self._timestamp))
        self._save(tcbs)
        return True
    
    
class PriorityRoundRobin:
    def __init__(self, quantum, timer, monitor):
        self._init(quantum, timer, monitor) #I thought I needed this in another place...

    def _init(self, quantum, timer, monitor):
        #configuration variables
        self._quantum = quantum #minimum atomic runtime of a job
        self._timer = timer #max time slice per job until timer interrupt
        self._timestamp = 0 #start time

        #bookkeeping
        self._ready_jobs_queue = ReadyQueue() #ReadyQueue of (priority, Queue<JCB>) tuples
        self._ready_tasks_set = set() #List of currently running tasks (i.e. tasks with uncompleted jobs) -> should be an ordered set (?)

        #saving
        self._monitor = monitor #monitor to log results

        #logging
        self._logger = loggingconfig.getLogger('SCH')

    def _run(self):
        """from current state, run simulation for one time quantum"""
        #fetch possibly ready new jobs from currently running tasks and append to queue
        for tcb in self._ready_tasks_set:
            while tcb.has_ready_job():
                next_jcb = tcb.next_job()
                next_jcb.job['release'] = self._timestamp
                self._ready_jobs_queue.put((tcb.task['priority'], next_jcb))
                self._logger.debug('ts {}: enqueued {}'.format(self._timestamp, next_jcb))

        try:
            #get highest priority job queue
            p, q = self._ready_jobs_queue.get_nowait()

            #get current job
            jcb = q.queue[0]

            if jcb.parentTCB.past_critical:
                self._logger.info('ts {}: {} -> parent {} is past critical. Skip.'.format(self._timestamp, jcb, jcb.parentTCB))
                jcb.log_success(False)
                if jcb.is_running:
                    jcb.log_interrupt()
                    jcb.parentTCB.log_interrupt()
                if jcb.is_youngest:
                    pass
                    #self._logger.info('ts {}: {} -> youngest. Kill. 1'.format(self._timestamp, jcb, jcb.parentTCB))
                    #self._monitor.task_finish(jcb.parentTCB.task)
                    #self._logger.info('ts {}: {} saved.'.format(self._timestamp, jcb.parentTCB))
                try:
                    q.get_nowait()
                except Empty:
                    self._logger.error('ts {}: This should never happen'.format(self._timestamp))
                if not q.empty():
                    self._ready_jobs_queue.put((p, q))
                return

            self._logger.debug('ts {}: Scheduling {}'.format(self._timestamp, jcb))

            #update state
            #if run for the first time, log start_date
            if not jcb.has_started:
                jcb.log_start_date(self._timestamp)

            #check for priority interrupt
            if jcb.is_running and jcb.had_interrupt(self._timestamp):
                jcb.log_interrupt()
                jcb.parentTCB.log_interrupt()
            #if not resuming execution in timeslice log activation
            if jcb.is_interrupted:
                jcb.log_activation(self._timestamp)
                jcb.parentTCB.log_activation(self._timestamp)

            #update state
            jcb.run(self._timestamp, self._quantum)
            jcb.parentTCB.run(self._timestamp, self._quantum)

            #case: this job is finished
            if jcb.has_finished:
                jcb.log_end_date(self._timestamp + self._quantum)
                jcb.log_interrupt()
                jcb.log_success(True)
                try:
                    q.get_nowait()
                except Empty:
                    self._logger.error('ts {}: This should never happen'.format(self._timestamp))

                self._logger.info('ts {}: {} is finished.'.format(self._timestamp, jcb))

                jcb.parentTCB.log_interrupt()
                #case: this job is finished AND it was the last job of this task
                if jcb.parentTCB.has_finished:
                    jcb.parentTCB.log_end_date(self._timestamp + self._quantum)
                    jcb.parentTCB.log_success(True)
                    self._ready_tasks_set.remove(jcb.parentTCB)
                    self._monitor.task_finish(jcb.parentTCB.task)
                    self._logger.info('ts {}: This was the last job -> {} is finished.'.format(self._timestamp, jcb.parentTCB))
                    self._logger.info('ts {}: {} saved.'.format(self._timestamp, jcb.parentTCB))

            #case: this job is not finished yet
            else:
                #check for due timer interrupt
                if jcb.due_timer_interrupt(self._timer):
                    #move to end of this priorities' queue
                    try:
                        q.get_nowait()
                        q.put(jcb)
                    except Empty:
                        self._logger.error('ts {}: This should never happen'.format(self._timestamp))

                    #log timer interrupt
                    jcb.log_interrupt()
                    jcb.parentTCB.log_interrupt()

                    self._logger.debug('ts {}: {} timer interrupt -> RR re-queue'.format(self._timestamp, jcb))

            #if this prio level's queue is not empty put tuple back into the readyqueue
            if not q.empty():
                self._ready_jobs_queue.put((p, q))

        except Empty:
            #if there are no jobs to be scheduled, just advance doing nothing
            #self._logger.info('ts {}: ReadyQueue is empty'.format(self._timestamp))
            pass

        #update all currently running tasks on the machine
        remove_set = []
        for tcb in self._ready_tasks_set:
            if not tcb.has_started:
                tcb.log_start_date(self._timestamp)
            tcb.live(self._quantum)
            if tcb.past_critical:
                tcb.log_critical_end_date()
                tcb.log_success(False)
                remove_set.append(tcb)
                self._monitor.task_finish(tcb.task)
                self._logger.info('ts {}: {} saved.'.format(self._timestamp, tcb))
                if not tcb.jcbs or tcb.youngest_jcb.has_finished:
                    pass
                self._logger.info('ts {}: {} is past critical. Kill. 2'.format(self._timestamp, tcb))

        for tcb in remove_set:
            self._ready_tasks_set.remove(tcb)

        #advance simulation one quantum
        self._timestamp += self._quantum

    def schedule(self, new_task):
        """expects a task with an release higher than all previous tasks,
        and advances the simulation until this task's release, s.th. it can be enqueued"""

        if new_task['release'] <= self._timestamp - self._quantum:
            raise ValueError('task must not start earlier than the current simulation timestamp')

        while self._timestamp < new_task['release']:
            self._run()

        self._logger.info('ts {}: Scheduling task {}...'.format(self._timestamp, new_task['id']))
        self._ready_tasks_set.add(TCB(new_task))
        self._logger.debug('ts {}: {} tasks in ReadyQueue'.format(self._timestamp, len(self._ready_tasks_set)))

    def finish(self):
        """runs the simulation until all current tasks are finished"""
        self._logger.info('ts {}: Finishing tasks...'.format(self._timestamp))
        while self._ready_tasks_set and not self._ready_jobs_queue.empty():
            self._run()

    def run_sample(self, sample, size):
        """simulate scheduling of a sample set of tasks"""
        if not isinstance(sample, Iterable):
            raise TypeError('sample must be iterable')

        if not sample:
            raise Exception('sample must not be empty')

        self._logger.info('ts {}: Sample Simulation Start.'.format(self._timestamp))
        for i in range(size):
            self.schedule(next(sample))

        self.finish()
        self._logger.info('ts {}: Sample Simulation Done.'.format(self._timestamp))
