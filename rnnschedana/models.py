"""LSTM model"""
from collections import namedtuple
from recordclass import recordclass
from pathlib import Path
from abc import ABC, abstractmethod

import tensorflow as tf

import preparedata as prepdata
import samples
from utilities import classproperty

tf.logging.set_verbosity(tf.logging.DEBUG)

FLAGS = tf.app.flags.FLAGS

FEATURE_ATTRIBUTES = [
'quota',
'pkg',
'priority',
'period',
'numberofjobs',
'offset',
'executiontime',
'criticaltime',
'deadline',
'success']

class Model(ABC):
    """Abstract base class for all models using the tf.estimator API"""
    #The tf.estimator.Estimator API is convenient, but I'm not happy with how it
    #draws the line between a class of models and an instance of a model. Some
    #lightweight OOP hopefully makes the code a little more comprehensible.

    #default values for hyper parameters this model expects
    DEF_HPARAMS = tf.contrib.training.HParams()
    @classproperty
    def HPARAMS(cls):
        return tf.contrib.training.HParams(**cls.DEF_HPARAMS.values())
    #default values for training parameters
    DEF_TPARAMS = tf.contrib.training.HParams()
    @classproperty
    def TPARAMS(cls):
        return tf.contrib.training.HParams(**cls.DEF_TPARAMS.values())

    def __init__(self, model_dir, run_config=None, hparams=None, tparams=None):
        #set default run_config
        if not run_config:
            run_config = tf.estimator.RunConfig(
                save_checkpoints_secs = 300,
                keep_checkpoint_max = 3) 
            
        #set default hyper parameters
        if hparams:
            self._hparams = hparams
        else:
            self._hparams = self.DEF_HPARAMS

        if tparams:
            self._tparams = tparams
        else:
            self._tparams = self.DEF_TPARAMS

        def model_fn(features, labels, mode):
            return self._model_fn(features, labels, mode)
        self._estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            config=run_config,
            params=None, #I handle this through encapsulation in these Model classes
            warm_start_from=None #TODO: read up
        )

    @property
    def estimator(self):
        return self._estimator

    @property
    def hparams(self):
        return self._hparams

    def set_hparams(self, hparams):
        self._update_hparams(self._hparams, hparams)

    @property
    def tparams(self):
        return self._tparams

    def set_tparams(self, tparams):
        self._update_hparams(self._hparams, tparams)

    @staticmethod
    def _update_hparams(hparams, params):
        """utility function"""
        if isinstance(params, dict):
            self._hparams.override_from_dict(params)
        elif isinstance(params, tf.contrib.training.HParams):
            self._hparams.override_from_dict(params.values())
        else:
            self._hparams.parse(params)

    @abstractmethod
    def _model_fn(self, features, labels, mode):
        """model_fn as required by tf.estimator.Estimator"""
        pass

    @abstractmethod
    def _train_input_fn(self):
        pass

    @abstractmethod
    def _eval_input_fn(self):
        pass

    #@abstractmethod
    #def _serve_input_fn(self):
    #    pass

    @abstractmethod
    def train_and_evaluate(self):
        pass

    #@abstractmethod
    #def get_description(self):
    #    pass

class BaseModel(Model):
    """Base Model Class for this project. Namely defining input signature -> What are the features?"""
    DEF_HPARAMS = Model.HPARAMS

    DEF_TPARAMS = Model.TPARAMS
    DEF_TPARAMS.add_hparam('train_data', None)
    DEF_TPARAMS.add_hparam('eval_data', None)
    DEF_TPARAMS.add_hparam('train_batch_size', 32)
    DEF_TPARAMS.add_hparam('eval_batch_size', 32)
    DEF_TPARAMS.add_hparam('train_steps', 1024)
    DEF_TPARAMS.add_hparam('eval_steps', 1024)
    DEF_TPARAMS.add_hparam('train_num_epochs', None)
    DEF_TPARAMS.add_hparam('eval_num_epochs', 1)
    DEF_TPARAMS.add_hparam('optimizer', tf.train.AdamOptimizer)

    #feature_columns define the model's input signature
    _FeatureColumnsSpec = recordclass('_FeatureColumnsSpec', FEATURE_ATTRIBUTES)
    DEF_FEATURE_COLUMNS = _FeatureColumnsSpec(
        quota=None,
        pkg=tf.feature_column.categorical_column_with_vocabulary_file('pkg', 'tasknames.txt', 8),
        priority=tf.feature_column.numeric_column('priority'),
        period=tf.feature_column.numeric_column('period'),
        numberofjobs=tf.feature_column.numeric_column('numberofjobs'),
        offset=tf.feature_column.numeric_column('offset'),
        executiontime=None,
        criticaltime=tf.feature_column.numeric_column('critical_time'),
        deadline=None, #not using this by default
        success=None #this is the lable
    )

    #@classproperty

    def __init__(self, model_dir, run_config=None, hparams=None, tparams=None, feature_columns=None):
        if feature_columns:
            self._feature_columns = feature_columns
        else:
            self._feature_columns = self.DEF_FEATURE_COLUMNS

        super().__init__(model_dir, run_config, hparams, tparams)

    @staticmethod
    def _parse_features(sequence_example):
        #There is a built-in function to infer this spec automatically for tf.train.Example,
        #however there is not yet a way to do this automatically for tf.train.SequenceExample
        context_feature_parse_spec = {
            'length' : tf.FixedLenFeature([], tf.int64)
        }

        sequence_feature_parse_spec = {
            'pkg' : tf.FixedLenSequenceFeature([], tf.string),
            'priority' : tf.FixedLenSequenceFeature([], tf.int64),
            'critical_time' : tf.FixedLenSequenceFeature([], tf.int64),
            'numberofjobs' : tf.FixedLenSequenceFeature([], tf.int64),
            'period' : tf.FixedLenSequenceFeature([], tf.int64),
            'offset' : tf.FixedLenSequenceFeature([], tf.int64),
            'success' : tf.FixedLenSequenceFeature([], tf.int64)
        }

        context, features = tf.parse_single_sequence_example(
            sequence_example,
            context_features=context_feature_parse_spec,
            sequence_features=sequence_feature_parse_spec
        )
        return context, features


    def _input_fn(self, data, batch_size, shuffle_buffer_size=1, num_epochs=None):
        #get dataset of examples from .tfrecord files
        dataset = prepdata.dataset_from_shuffled_tfrecords(data)

        #batch_size batches per element
        dataset = dataset.batch(batch_size)

        #shuffle
        if shuffle_buffer_size > 1:
            dataset.shuffle(shuffle_buffer_size)

        #number epochs
        if num_epochs:
            dataset.repeat(num_epochs) #repeat for num_epochs
        else:
            dataset.repeat() #repeat indefinitely

        #TODO: batch padding ? otherwise every example needs to have the same length in time steps...

        #parse features as tensors from tf.train.Example/SequenceExample in the .tfrecord
        dataset.map(self._parse_features, num_parallel_calls=5)

        #create iterator
        iterator = dataset.make_one_shot_iterator()
        context, features = iterator.get_next()
        labels = features['success']
        features.update(context)
        return features, labels

    def _train_input_fn(self):
        return self._input_fn(
            data=self._tparams.train_data,
            batch_size=self._tparams.train_batch_size,
            num_epochs=self._tparams.train_num_epochs
        )

    def _eval_input_fn(self):
        return self._input_fn(
            data=self._tparams.evaluation_data,
            batch_size=self._tparams.eval_batch_size,
            num_epochs=self._tparams.eval_num_epochs
        )

    def train_and_evaluate(self):
        train_spec = tf.estimator.TrainSpec(
            input_fn=self._train_input_fn(),
            max_steps=self._tparams.train_steps,
            hooks=None
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=self._eval_input_fn(),
            steps=self._tparams.eval_steps,
            name=None,
            hooks=None,
            exporters=None,
            start_delay_secs=120,
            throttle_secs=600
        )
        tf.estimator.train_and_evaluate(
            self._estimator,
            train_spec,
            eval_spec
        )

class TF_LSTM(BaseModel):
    DEF_HPARAMS = BaseModel.HPARAMS
    DEF_HPARAMS.add_hparam('num_units', [64])
    DEF_HPARAMS.add_hparam('use_peepholes', [False])
    DEF_HPARAMS.add_hparam('initializer', [tf.initializers.ones])
    DEF_HPARAMS.add_hparam('forget_bias', [1.0])
    DEF_HPARAMS.add_hparam('activation', [tf.tanh])

    DEF_TPARAMS = BaseModel.TPARAMS

    def _model_fn(self, features, labels, mode):
        #bucketize priority column
        #self._feature_columns.priority = \
        #    tf.feature_column.bucketized_column(self._feature_columns['priority'], 8)
        #create an input layer from feature_columns
        input_layer = tf.feature_column.input_layer(
            features=features,
            feature_columns=[c for c in self._feature_columns if c],
            weight_collections=['input_layer_variables'],
            trainable=True)

        #define layers of lstm_cells
        cells = []
        for n_u, u_p, init, f_b, act in zip(
            self._hparams.num_units,
            self._hparams.use_peepholes,
            self._hparams.initializer,
            self._hparams.forget_bias,
            self._hparams.activation
            ):

            cell = LSTMCell(
                num_units = n_u, #width of the model -> 'memory/state size'
                use_peepholes=u_p, #TODO: have to read up on this again
                cell_clip=None, #leaving this alone for now
                initializer=init, #initialize with ones by default
                num_proj=None, #TODO: huh?
                proj_clip=None, #maybe experiment with this
                num_unit_shards=None, #deprecated
                num_proj_shards=None, #deprecated... I love TensorFlow
                forget_bias=f_b, #forget bias initialized with ones by default
                state_is_tuple=True, #(output, state of type LSTMStateTuple) -> (output, (state, output))
                #...this implementation is so weird, why not just use ONE tuple (state, output)...
                activation=act, #activation function -> tanh default
                reuse=None) #every cell has its own set of weights
            cells.append(cell)

        #chain lstm layers and merge into a multi cell
        #this changes the (output, state) tuple structure! -> ...
        #...with n = len(cells)
        #...output has shape [train_batch_size, seq_example_length, n]
        #...state is now a n-tuple of LSTMStateTuple s ((state_0, output_0), ..., (state_n, output_n))
        multi_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

        #get initial state, this should account for the changed tuple structure
        #i.e. a n-tuple of LSTMStateTuples
        initial_state = multi_cell.zero_state(batch_size, dtype=tf.float32)

        #unroll dynamically
        output, _ = tf.nn.dynamic_rnn( #just need the final output
            cell=multi_cell, #this should propagate the time_steps unroll through all layers...
            inputs=input_layer, #input tensor
            sequence_length=None, #redundant when using dynamic_rnn isn't it?
            initial_state=initial_state,
            dtype=None, #inferred
            parallel_iterations=None, #TODO: read up
            wap_memory=False, #TODO: read up
            time_major=False, #data comes in batch major format -> [batch_size, time_steps, ...]
            scope=None) #default

        #aggregate output to a single logit
        logit = tf.layers.dense(output, 1)

        #get probability
        prob = tf.sigmoid(logit)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'prob' : prob}
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                prediction_hooks=None
            )

        assert mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.TRAIN
        #if we're not predicting we're evaluating or training...

        #we want to maximize cross entropy across each batch => comes down to simple log_loss
        loss = tf.reduce_mean(tf.losses.log_loss(
            labels,
            prob,
            weights=1.0,
            scope=None,
            loss_collection=ops.GraphKeys.LOSSES,
            reduction=Reduction.SUM_BY_NONZERO_WEIGHTS))

        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(labels, prob)
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops,
                evaluation_hooks=None
            )

        assert mode == tf.estimator.ModeKeys.TRAIN

        #optimzation method
        optimizer = self._tparams.optimizer()

        #each step minimize the loss according to the selected optimization method
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            export_outputs=None,
            training_chief_hooks=None,
            training_hooks=None,
            scaffold=None
        )
