# -*- coding: utf-8 -*-
#Baustelle
"""(Command Line) Interface script for rnnschedana."""
from datetime import datetime

import datageneration as datagen
from mongointerface import MongoInterface
import preparedata as prepdata
import loggingconfig
import models
import samples
import utilities
import pprint


logger = loggingconfig.getLogger('INTERFACE')

TF_RECORDS_DIR = '/home/hans/Documents/Uni/TUM/Informatik/WS18/BA/tfrecords'

def create_data(experiment_spec, tfrecords_dir=TF_RECORDS_DIR, shard_size=1000, parallel=True, overwrite=True):
    sw = utilities.StopWatch('total')
    mongo = MongoInterface()

    try:
        doc = mongo.get_experiment_doc_by_name(experiment_spec.name)
        if not overwrite:
            raise Exception('An experiment with this name exists already.')
        else:
            mongo.delete_experiment_by_name(experiment_spec.name)
    except Exception:
        pass

    logger.info('Register experiment...')
    doc = mongo.register_experiment(experiment_spec)
    #start simulation and fill experiment with samples
    logger.info('Commence simulation...')
    sw.start('simulate')
    if parallel:
        datagen.simulate_parallel(experiment_spec)
    else:
        datagen.simulate(experiment_spec)
    sw.stop('simulate')
    #sort and save samples in experiment
    logger.info('Create experiment collection...')
    sw.start('summarize')
    mongo.sample_colls_to_experiment_coll(experiment_spec.coll_name)
    sw.stop('summarize')
    logger.info('Save as tfrecords...')
    sw.start('serialize')
    mongo.experiment_coll_to_tfrecords(experiment_spec.coll_name, tfrecords_dir, shard_size=1000)
    sw.stop('serialize')

    logger.info('\nCompleted data generation for:\n {}'.format(pprint.pformat(doc)))
    swlogs = sw.logs
    logger.info("""\nRunning Times:
    {0:>10s}: {1}
    {2:>10s}: {3}
    {4:>10s}: {5} (Sort and Summarize in Experiment Collection)
    {6:>10s}: {7} (Save as .tfrecord files)""".format(
    'total', swlogs['total']['total'],
    'simulate', swlogs['simulate']['total'],
    'summarize', swlogs['summarize']['total'],
    'serialize', swlogs['serialize']['total']))
    mongo.update_experiment_doc(experiment_spec.id, swlogs)


MODEL_BASE_DIR = '/home/hans/Documents/Uni/TUM/Informatik/WS18/BA/models'
HANS_CLUSTER = ''

def train():
    pass