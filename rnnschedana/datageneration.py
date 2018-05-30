import logging
from multiprocessing import Pool
from collections import namedtuple
from hashlib import sha1
from os import getpid
from datetime import datetime
import scipy

from taskgen.distributor import Distributor
from taskgen.sessions.simso import SimSoSession
from taskgen.sessions.file import FileSession
from taskgen.monitors.stdout import StdOutMonitor

import samples
import loggingconfig
from mongomonitor import MongoMonitor
from mongointerface import MongoInterface
from schedulerproxy import PriorityRoundRobin
from schedulerproxy import PriorityRoundRobin2
import utilities

logger = loggingconfig.getLogger('DATAGEN')

BaseSpec = namedtuple('BaseSpec', 'name sample_rv_class sample_length size sch_q sch_it description id') #do not use like this, use ExperimentSpec
class ExperimentSpec(BaseSpec):
    def __new__(cls, name, sample_rv_class, sample_length, size, sch_q, sch_it, description, id=None):
        #This is called for every process during multiprocessing with the attributes defined in
        #the PARENT process.
        #Hence it is called not with id=None, but with the id=<hash of the parent process' ExperimentSpec>
        #This results in a new id for the child processes which messes up the collection dependencies.
        #-> always set id to 0 first and then hash the id from the attributes again!
        tmp = BaseSpec.__new__(cls, name, sample_rv_class, sample_length, size, sch_q, sch_it, description, 0)
        id = sha1(''.join(repr(tmp)).encode('utf-8')).hexdigest()
        return BaseSpec.__new__(cls, name, sample_rv_class, sample_length, size, sch_q, sch_it, description, id)
    @property
    def coll_name(self):
        return '{}.{}'.format(self.name, self.id)

ES_MINI = ExperimentSpec(
    name='mini', #name of the simulation
    size=100, #number of simulations
    sample_rv_class=samples.MC_0, #rv class to generate samples
    sample_length=100, #maximum length of each simulation
    sch_q=1, #time quantum of scheduler >= 1
    sch_it=10,
    description='Mini Example Spec')

ES_EXAMPLE = ExperimentSpec(
    name='test', #name of the simulation
    size=10, #number of simulations
    sample_rv_class=samples.MC_0, #rv class to generate samples
    sample_length=100, #maximum length of each simulation
    sch_q=1, #time quantum of scheduler >= 1
    sch_it=10,
    description='Example Spec to for illustration')

ES_DEBUG = ExperimentSpec(
    name='debug', #name of the simulation
    size=10, #number of simulations
    sample_rv_class=samples.MC_0, #rv class to generate samples
    sample_length=100, #length of each simulation
    sch_q=1, #time quantum of scheduler >= 1
    sch_it=20,
    description='Test Spec to assess problems and bottle-necks'
)

def sim_step(enumerated_experiment_spec):
    scipy.random.seed(int(time.time()))
    sample_id = enumerated_experiment_spec[0]
    experiment_spec = enumerated_experiment_spec[1]
    logger.info('Commence sim_step {}/{} on PID {}'.format(sample_id+1, experiment_spec.size, getpid()))
    sw = utilities.StopWatch('sim_step')
    sw.start()
    coll_name = '{}.{}_{}'.format(experiment_spec.coll_name, experiment_spec.sample_rv_class, sample_id)
    mongo = MongoInterface()
    sample_coll = mongo.get_coll(coll_name, index='id')
    monitor = MongoMonitor(sample_coll)
    scheduler = PriorityRoundRobin2(experiment_spec.sch_q, experiment_spec.sch_it, monitor)
    rv = experiment_spec.sample_rv_class()
    scheduler.run_sample(rv, experiment_spec.sample_length)
    dt = sw.stop()
    logger.info('Completed sim_step {}/{} after {}'.format(sample_id+1, experiment_spec.size, dt))
    return True

def simulate_parallel(experiment_spec):
    pool = Pool()
    sims = ((c, experiment_spec) for c in range(experiment_spec.size))
    pool.map(sim_step, sims)
    pool.close()
    pool.join()

def simulate(experiment_spec):
    sims = ((c, experiment_spec) for c in range(experiment_spec.size))
    for sim in sims:
        sim_step(sim)

def main():
    simulate_parallel(ES_EXAMPLE)

if __name__ == '__main__':
    main()
