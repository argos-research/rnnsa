import tensorflow as tf

import utilities

def parse_feature(raw_feature):
    if not isinstance(raw_feature, list):
        raw_feature = list(raw_feature)

    feature = tf.train.Feature()

    #the value field of int64_list, float_list, bytes_list is of the scalar type int64, float, bytes
    #so we have to use append()/extend()
    if isinstance(raw_feature[0], int):
        return feature.int64_list.value.extend(raw_feature)
    elif isinstance(raw_feature[0], float):
        feature.float_list.value.extend(raw_feature)
    else:
        feature.bytes_list.value.extend(list(map(utilities.to_bytes, raw_feature)))

    return feature

def parse_feature_list(raw_feature_iterator):
    #the feature field of a feature_list is a message type, so we have to to use add() or extend()
    feature_iterator = map(parse_feature, raw_feature_iterator)
    feature_list = tf.train.FeatureList()

    feature_list.extend(list(feature_iterator))
    return feature_list

def parse_sequence_example(raw_feature_iterator_dict, raw_context_feature_dict=None):
    """Given dictionaries of
    <feature_name>:<raw_feature_iterator>,
    <label_name>:<raw_label_iterator>,
    <feature_name>:<raw_context_feature>,
    this function parses a tf.train.SequenceExample protocol buffer"""
    seqex = tf.train.SequenceExample()

    #save context features
    if not raw_context_feature_dict == None:
        for key, value in raw_context_feature_dict:
            #Because of how protocol buffers work, the only way to assign a child message field
            #is to either go all the way down to a primitve field of the last child message OR
            #copy the content from a message field from the same type. Yes, this IS ridiculous...
            #This means we can't assign a new feature in the context directly, but we can create a new feature...
            feature = seqex.context.feature[key]
            #...and copy the contents from the parsed raw feature. -> lots of overhead
            feature.CopyFrom(parse_feature(value))

    for key, value in raw_feature_iterator_dict:
        #create a new feature_list
        feature_list = seqex.feature_lists.feature_list[key]
        #copy contents from parsed feature_list
        feature_list.CopyFrom(parse_feature_list(value))

    return seqex

#Different way of doing this:----------------------------------------------------------------------------

def assign_raw_feature_to_feature(raw_feature, feature):
    if not isinstance(raw_feature, list):
        raw_feature = [raw_feature]

    #the value field of int64_list, float_list, bytes_list is of the scalar type int64, float, bytes
    #so we have to use append()/extend()
    if isinstance(raw_feature[0], int):
        return feature.int64_list.value.extend(raw_feature)
    elif isinstance(raw_feature[0], float):
        feature.float_list.value.extend(raw_feature)
    else:
        feature.bytes_list.value.extend(list(map(utilities.to_bytes, raw_feature)))

def assign_raw_feature_iterator_to_feature_list(raw_feature_iterator, feature_list):
    #the feature field of a feature_list is a message type, so we have to to use add() or extend()
    for raw_feature in raw_feature_iterator:
        feature = feature_list.feature.add()
        assign_raw_feature_to_feature(raw_feature, feature)

def parse_sequence_example_2(raw_feature_iterator_dict, raw_context_feature_dict=None):
    """Alternative implementation of parse_sequence_example, which should consume less time and memory,
    because we don't create everything twice and move data around. It's just harder to understand in my opinion."""
    seqex = tf.train.SequenceExample()

    if not raw_context_feature_dict == None:
        for key, value in raw_context_feature_dict.items():
            feature = seqex.context.feature[key]
            assign_raw_feature_to_feature(value, feature)

    for key, value in raw_feature_iterator_dict.items():
        feature_list = seqex.feature_lists.feature_list[key]
        assign_raw_feature_iterator_to_feature_list(value, feature_list)

        return seqex

def dataset_from_shuffled_tfrecords(data):
    #create a dataset with all file_paths
    file_paths = tf.matching_files(data)
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)

    #shuffle the .tfrecord file_paths
    num_paths = tf.shape(file_paths, out_type=tf.int64)[0]
    dataset.shuffle(num_paths)

    #map to the actual data in the .tfrecord files/shards
    circle_length = num_paths
    dataset.interleave(
        tf.data.TFRecordDataset,
        circle_length)

    return dataset

def split_sequences_from_context(sample_dict):
    """Separates context features from feature sequences"""
    feature_iterator_dict = sample_dict
    context_feature_dict = None
    for key, value in feature_iterator_dict.items():
        if not isinstance(value, list):
            if not context_feature_dict:
                context_feature_dict = {}
            context_feature_dict.update({key : value})
    for key in context_feature_dict.keys():
        del feature_iterator_dict[key]
    return feature_iterator_dict, context_feature_dict
