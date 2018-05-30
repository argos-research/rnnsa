"""Module for training data/sample generation"""
import re
import numpy as np
import scipy.stats as ss
import time
import math
from collections import Iterable
from collections import namedtuple
import pprint as pp
import itertools
from taskgen.taskset import TaskSet
from taskgen.taskset import BlockTaskSet
from taskgen.task import Task


def RV(dist, lbound=None, ubound=None):
    """Base generator for any random variable, specified by a frozen
    scipy.stats distribution"""
    if lbound or ubound:
        while True:
            yield int(np.clip(dist.rvs(size=1), lbound, ubound))
    else:
        while True:
            yield int(dist.rvs(size=1))

def RW(dist, start):
    """Base generator for any random walk, specified by a frozen
    scipy.stats distribution."""
    t = start
    while True:
        t += int(dist.rvs(size=1))
        yield t

def PoissonProcessRW(beta, start=0):
    """Generator for poisson process arrivals with rate 1/beta mean inter-arrival time"""
    return RW(ss.expon(loc=0, scale=beta), start)

def NormalRV(mean, vol, min=None, max=None):
    """Generator for normal distributed RVs"""
    return RV(ss.norm(loc=mean, scale=vol), min, max)

def UniformRV(low, high):
    """Generator for uniformly distributed RVs"""
    return RV(ss.randint(low, high))

class PkgRV:
    """Generator Class generating random tasknames."""
    def __init__(self, tasknames_abs_freq_dict=None):
        if not tasknames_abs_freq_dict:
            tasknames_abs_freq_dict = {
                'cond_42' : 125,
                'cond_mod' : 125,
                'hey' : 125,
                'idle' : 125,
                'linpack' : 125,
                'namaste' : 125,
                'pi' : 125,
                'tumatmul' : 125
            }

        self.tasknames = []
        self.abs_freqs = []
        for key, value in tasknames_abs_freq_dict.items():
            self.tasknames.append(key)
            self.abs_freqs.append(value)

        n = float(sum(self.abs_freqs))
        probs = list(map(lambda x: x/n, self.abs_freqs))
        self.pkgrv = ss.multinomial(1, probs)

    def sample(self):
        index_array = self.pkgrv.rvs(size=1, random_state=None)
        i = np.asscalar(np.where(index_array==1)[1])
        return self.tasknames[i]

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample()

class ExecutiontimeRVs:
    """Encapsulates RVs to be used for execution times of any given pkg."""
    def __init__(self, dist=None):
        self.dist = {
            'cond_42' : UniformRV(low=10, high=100),
            'cond_mod' : UniformRV(low=10, high=100),
            'hey' : UniformRV(low=10, high=100),
            'idle' : UniformRV(low=10, high=100),
            'linpack' : UniformRV(low=10, high=100),
            'namaste' : UniformRV(low=10, high=100),
            'pi' : UniformRV(low=10, high=100),
            'tumatmul' : UniformRV(low=10, high=100)
        }
        if dist:
            self.dist.update(dist)

    def sample(self, pkg):
        """Returns executiontime random sample for this task"""
        return next(self.dist[pkg])

class SampleRVMeta(type):
    def __str__(self):
        return  self.__name__

class SampleRV(object, metaclass=SampleRVMeta):
    """Generator, generating tasks according to attribute generators in attribute_generator_dict"""

    def __init__(self, attribute_generator_dict=None, etrvs=ExecutiontimeRVs()):
        #default attributes
        self._attributes = {
            'id': itertools.count(),
            'config': {},
            'quota': '1M',
            'pkg': 'hey',
            'priority': 63,
            'period': 10,
            'numberofjobs': 1,
            'offset': 0,
            'executiontime': 100,
            'criticaltime': 120,
            'deadline': None,
            'release':None
        }
        if attribute_generator_dict:
            self._attributes.update(attribute_generator_dict)
        self.etrvs = etrvs
        self.last_release = None

    def post_generation(self, task):
        """Overwrite in subclass to do something more fancy than 2x exectime.
        Accepts a task and sets new values for an attribute as a function of other task attributes,
        e.g. a new task['criticaltime'] as a function of task['executiontime']"""
        task['executiontime'] = self.etrvs.sample(task['pkg'])
        task['criticaltime'] = 4 * task['executiontime']
        if self.last_release:
            task['offset'] = task['release'] - self.last_release
        else:
            task['offset'] = task['release']
        self.last_release = task['release']

    def __iter__(self):
        return self

    def __next__(self):
        task = Task({key: SampleRV._next_call_value(value) for key, value in self._attributes.items()})
        self.post_generation(task)
        return task


    @staticmethod
    def _next_call_value(obj):
        """a function that returns:
        - None if None
        - the next item of a generator if a generator is passed,
        - return value of a function if a function is passed,
        - obj if none of the above
        """
        if not obj:
            return obj
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            return obj
        try:
            iter(obj)
        except TypeError:
            #not iterable
            if callable(obj):
                return obj()
            else:
                return obj
        else:
            #iterable
            return next(obj)

#Same thing as Sample class but I can't inherit it
def task_generator(attributes):
    while True:
        yield Task({key: SampleRV._next_call_value(value) for key, value in attributes.items()})

class MC_0(SampleRV):
    def __init__(self):
        super().__init__({
            'pkg': PkgRV(),
            'priority': NormalRV(mean=63, vol=32, min=0, max=127),
            'period': UniformRV(low=10, high=20),
            'numberofjobs': UniformRV(low=1, high=10),
            'release': PoissonProcessRW(beta=10)
        })

class MC_1(SampleRV):
    def __init__(self):
        super().__init__({
            'pkg': PkgRV(),
            'priority': NormalRV(mean=63, vol=32, min=0, max=127),
            'period': NormalRV(10, 4, min=1),
            'numberofjobs': NormalRV(mean=5, vol=2, min=1),
            'release': PoissonProcessRW(50)
    })

class MC_2(SampleRV):
    def __init__(self):
        super().__init__({
            'pkg': PkgRV(),
            'priority': NormalRV(mean=63, vol=32, min=0, max=127),
            'period': NormalRV(10, 4, min=1),
            'numberofjobs': NormalRV(mean=5, vol=2, min=1),
            'release': PoissonProcessRW(50)
            })

    def post_generation(self, task):
        task['criticaltime'] = 32 * task['executiontime']
