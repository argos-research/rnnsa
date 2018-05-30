import pymongo
from pymongo import MongoClient
import uuid
import hashlib
import pandas as pd
import tensorflow as tf
from pathlib import Path
import itertools
import pprint
import preparedata
import utilities
import models
import samples

class MongoInterface:
    def __init__(self, database='rnnschedana', uri='127.0.0.1:27017'):
        client = MongoClient(uri)
        self._database = client[database]
        self._experiment_docs_coll = self.get_coll('experiment_docs', index='id')

    def get_coll(self, name, index=None, drop=False):
        coll = self._database[name]
        if index:
            coll.create_index(index, unique=True)
        if drop:
            coll.drop()
        return coll

    def drop_coll(self, name):
        self.get_coll(name, index=None, drop=True)

    def drop_all(self):
        for coll in self.collections():
            self.get_coll(coll, drop=True)

    def experiments(self):
        return list(self._experiment_docs_coll.find())
    
    def experiment(self, name, limit=0):
        return self.list_coll(self.get_experiment_coll_name_by_name(name), limit)
    
    def get_sample_coll_name(self, experiment_name, sample_id):
        exp_doc = self.get_experiment_doc_by_name(experiment_name)
        return '{}.{}_{}'.format(exp_doc['coll_name'], exp_doc['sample_rv_class'], sample_id)
    
    def get_sample_coll(self, experiment_name, sample_id):
        return self.get_coll(self.get_sample_coll_name(experiment_name, sample_id))
    
    def sample(self, experiment_name, sample_id):
        return list(self.get_sample_coll(experiment_name, sample_id).find().sort([('release', 1)]))

    def update_experiment_doc(self, id, fields):
        new_doc = self._experiment_docs_coll.find_one_and_update(
            {'id':id}, {'$set':fields}, upsert=False, return_document=pymongo.ReturnDocument.AFTER
        )
        return new_doc

    def list_coll(self, name, limit=0):
        return list(self.get_coll(name).find().limit(limit))

    def collections(self):
        return self._database.collection_names()

    def register_experiment(self, experiment_spec):
        experiment_doc = {key : str(value) for key, value in zip(experiment_spec._fields, list(experiment_spec))}
        experiment_doc.update({'coll_name' : experiment_spec.coll_name})
        self._experiment_docs_coll.insert_one(experiment_doc)
        self.get_coll(experiment_doc['coll_name'], index='id')
        return experiment_doc

    def get_experiment_doc_by_name(self, name, as_list=False):
        cursor = self._experiment_docs_coll.find({'name' : name})
        if cursor.count() < 1:
            raise Exception('There is no experiment doc with name {}'.format(name))
        docs = list(cursor)
        if as_list:
            return docs
        return docs[0]

    def get_experiment_doc_by_id(self, id):
        cursor = self._experiment_docs_coll.find_one({'id' : id})
        if not cursor:
            raise Exception('There is no experiment doc with id {}'.format(id))
        return cursor

    def get_epxeriment_id_by_name(self, name):
        experiment_doc = self.get_experiment_doc_by_name(name)[0]
        return experiment_doc['id']

    def get_experiment_coll_name_by_name(self, name):
        experiment_doc = self.get_experiment_doc_by_name(name)
        return experiment_doc['coll_name']

    def get_experiment_coll_by_name(self, name):
        return self.get_coll(self.get_experiment_coll_name_by_name(name))

    def get_experiment_coll_name_by_id(self, id):
        experiment_doc = self.get_experiment_doc_by_id(id)
        return experiment_doc['coll_name']

    def drop_experiment_sample_colls(self, experiment_coll_name):
        experiment_coll = self.get_coll(experiment_coll_name)
        for coll_name in self.collections():
            split_name = tuple(coll_name.split('.'))
            if len(split_name) == 3:
                exp_name, exp_id, sample_name = split_name
                if exp_name + '.' + exp_id == experiment_coll_name:
                    self.drop_coll(coll_name)

    def drop_experiment_coll_by_name(self, name):
        coll_name = self.get_experiment_coll_name_by_name(name)
        self.drop_coll(coll_name)
        self._experiment_docs_coll.delete_one({'name': name})
        return coll_name

    def drop_experiment_coll_by_id(self, id):
        coll_name = self.get_experiment_coll_name_by_id(id)
        self.drop_coll()
        self._experiment_docs_coll.delete_one({'id': id})
        return coll_name

    def delete_experiment_by_name(self, name):
        coll_name = self.drop_experiment_coll_by_name(name)
        self.drop_experiment_sample_colls(coll_name)

    def delete_experiment_by_id(self, id):
        coll_name = self.get_experiment_coll_name_by_id(id)
        self.drop_experiment_coll_by_id(id)
        self.drop_experiment_sample_colls(coll_name)

    def sample_observations(self, coll_name, features, sort_keys=[('release', 1)]):
        sample_coll = self.get_coll(coll_name)
        project = {'_id': False}
        for key in features:
            project.update({key : True})
        return sample_coll.find(projection=project).sort(sort_keys)

    def sample_colls_to_experiment_coll(
        self,
        experiment_coll_name,
        features=models.FEATURE_ATTRIBUTES,
        delete=False,
        sort_keys=[('release', 1)]
        ):
        experiment_coll = self.get_coll(experiment_coll_name)
        coll_names = self._database.collection_names()
        for coll_name in coll_names:
            split_name = tuple(coll_name.split('.'))
            if len(split_name) == 3:
                exp_name, exp_id, sample_name = split_name
                if exp_name + '.' + exp_id == experiment_coll_name:
                    cursor = self.sample_observations(coll_name, features, sort_keys)
                    sample_observations = list(cursor)
                    sample_dict = {key : [] for key in features}
                    sample_dict.update({'id' : sample_name})
                    for observation in sample_observations:
                        for key in features:
                            sample_dict[key].append(observation[key])
                    experiment_coll.insert_one(sample_dict)
        if delete:
            self.drop_experiment_sample_colls(experiment_coll_name)

    def experiment_coll_to_tfrecords(self, experiment_coll_name, tfrecords_dir, shard_size=100):
        #get experiment collection
        experiment_coll = self.get_coll(experiment_coll_name)
        #get sample cursor
        cursor = experiment_coll.find()

        #make direcotry for .tfrecord files
        dir_path = Path(tfrecords_dir)
        if not dir_path.exists():
            dir_path.mkdir()

        from_index = 0

        while True:
            try:
                sample_dict = next(cursor)
                to_index = from_index + shard_size - 1
                file_name = '{}.{}-{}.tfrecord'.format(experiment_coll_name, from_index, to_index)
                file_path = dir_path / file_name
                with tf.python_io.TFRecordWriter(str(file_path), options=None) as writer:
                    for _ in range(shard_size):
                        feature_iterator_dict, context_feature_dict = \
                            preparedata.split_sequences_from_context(sample_dict)
                        del context_feature_dict['_id']
                        seqex = \
                            preparedata.parse_sequence_example_2(feature_iterator_dict,
                                context_feature_dict)
                        writer.write(seqex.SerializeToString())
                        try:
                            sample_dict = next(cursor)
                        except StopIteration:
                            break
                from_index = to_index + 1
            except StopIteration:
                break

    def sample_to_df(self, experiment_name, sample_id, limit=0):
        sample_coll = self.get_sample_coll(experiment_name, sample_id)
        project = {
            '_id': False,
            'jobs': False,
            'config': False
        }
        cursor = sample_coll.find(projection=project, limit=limit).sort([('release', 1)])
        return pd.DataFrame(list(cursor))

    def experiment_to_df(self, name, limit=0):
        coll = self.get_experiment_coll_by_name(name)
        project = {
            '_id': False,
        }
        cursor = coll.find(projection=project, limit=limit)
        data = list(cursor)
        sample_dfs = []
        for sample in data:
            sample_length = len(sample['pkg'])
            sample['task_id'] = list(range(sample_length))
            df = pd.DataFrame(data=sample)
            df.set_index(['id', 'task_id'], inplace=True)
            sample_dfs.append(df)
        df = pd.concat(sample_dfs)
        return df

    def sample_jobs_to_df(self, sample, limit=0):
        #OUT OF ORDER
        sample_coll = self._database[sample]
        project = {
        '_id': False,
        'id': True,
        'jobs' : True
        }
        cursor = sample_coll.find(projection=project, limit=limit)

        def unwind_jobs(task):
            """moves all of a task's attributes down into each of its jobs,
            and returns the updated jobs list"""
            for job in task['jobs']:
                for attr in task.keys():
                    if attr != 'jobs':
                        job['task_{}'.format(attr)] = task[attr]
                        return task['jobs']

        jobs = utilities.flatten(map(unwind_jobs, cursor), ltypes=(list, tuple))
        jobs_df = pd.DataFrame(jobs)
        return jobs_df
