# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""
#######################################
# Rename to SENTIJ as the processor (Japanese Version)
# Define labels as 0,1 which read from file label.tsv (reduce labels from EN version)
# Read text from column 2, read label from column 0, renumber guid
# pass text to text_a of class InputExample
# set default parameter so it can be run in local IDE easily(CPU)
#
# fine-tune with custom code:
#   pool layer(768) -> dense layer(32) -> output layer(4)
# ipynb(bert5) is used to prepare finetune on colab TPU  --TBD
# model file saved in gs, directory SENTIJ/model.ckpt-****.data-04999-of-00001
# Additional information printing: output layer info of classic model
#
# Now change the inter-thread communication to inter-process communication
#######################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
#import bert_master.modeling
#import bert_master.optimization
#import bert_master.tokenization
import modeling
import optimization
import tokenization
import tensorflow as tf

import multiprocessing
import time
import sys

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", "./data",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", "./bert_config_multi.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "sentij", "The name of the task to train.")

flags.DEFINE_string("vocab_file", "./vocab_multi.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "./model",
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

# My parameter
flags.DEFINE_string(
    "subversion", None,
    "The output directory subname where the model checkpoints will be written.")

flags.DEFINE_bool(
    "alignepochs", False,
    "The output directory subname where the model checkpoints will be written.")
# End of my parameter

# BSMS start
flags.DEFINE_bool(
    "parallel", True, 
    "Whether to run estimator prediction in another process.")
    
flags.DEFINE_bool(
    "verbose", True, 
    "Whether to print info for parallel logic.")

flags.DEFINE_bool(
    "silent", False, 
    "Whether to print info for bert master.")
# BSMS end

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 2, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class SentiJProcessor(DataProcessor):
  """Processor for the Sentimental data set (GLUE version)."""
  def __init__(self):
    self.label_list = None
    self.tokenizer = None

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir, data_file):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, data_file)), "test")

  def get_test_examples_from_list(self, input_list):
    """See base class."""
    templist = ["dummyheader"]
    templist.extend(input_list)
    return self._create_examples([["",str(i),l] for (i,l) in enumerate(templist)], "test")

  def get_labels(self, data_dir):
    """See base class."""
    labels = self._read_tsv(os.path.join(data_dir, "label.tsv"))
    return labels[0]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      if len(line) < 2:
        continue
      guid = "%s-%s" % (set_type, i)
      # text_a = tokenization.convert_to_unicode(line[3])
      # text_b = tokenization.convert_to_unicode(line[4])
      text_a = tokenization.convert_to_unicode(line[2])
      text_b = None
      if set_type == "test":
        label = "0"
      else:
        label = tokenization.convert_to_unicode(line[0])
      if i%100 == 0:
        if not FLAGS.silent:
            tf.logging.info("%s No.%s (%s) %s" % (set_type, line[1], label, text_a[:20]))
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    if not FLAGS.silent:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value
  #MyInfo print
  #if not FLAGS.silent:
  #  tf.logging.info(">>>>MyInformaiton: output pooler layer size = %d ",hidden_size)
  #end
	
  #Define my own logic for model
  #Create variables for 1 addition dense layer
  hidden_size2 = 32
  #output_weights = tf.get_variable(
  #    "output_weights", [num_labels, hidden_size],
  #    initializer=tf.truncated_normal_initializer(stddev=0.02))
  #
  #output_bias = tf.get_variable(
  #    "output_bias", [num_labels], initializer=tf.zeros_initializer())
  dense_weights_1 = tf.get_variable(
      "dense_weights_1", [hidden_size2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  dense_bias_1 = tf.get_variable(
      "dense_bias_1", [hidden_size2], initializer=tf.zeros_initializer())
      
  output_weights_2 = tf.get_variable(
      "output_weights_2", [num_labels, hidden_size2],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias_2 = tf.get_variable(
      "output_bias_2", [num_labels], initializer=tf.zeros_initializer())
  #end

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    #Additional dense layer
    #logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    #logits = tf.nn.bias_add(logits, output_bias)
    dense_1 = tf.matmul(output_layer, dense_weights_1, transpose_b=True)
    dense_1 = tf.nn.bias_add(dense_1, dense_bias_1)
    if is_training:
      # I.e., 0.1 dropout
      dense_1 = tf.nn.dropout(dense_1, keep_prob=0.9)

    logits = tf.matmul(dense_1, output_weights_2, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias_2)
    #end
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    if not FLAGS.silent:
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
          tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if not FLAGS.silent:
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
          init_string = ""
          if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
          #tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
          #                init_string)
          tf.logging.info("  name = %s, shape = %s, trainable=%r %s", var.name, var.shape, var.trainable,
                          init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn

# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      if not FLAGS.silent:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features

#--------------------- My code begin -------------------------#
def build_bert_estimator():

  processors = {
      "sentij": SentiJProcessor
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  processor.label_list = processor.get_labels(FLAGS.data_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    # set save_checkpoints_steps to be consistent with epochs
    if FLAGS.alignepochs:
      FLAGS.save_checkpoints_steps = int(len(train_examples) / FLAGS.train_batch_size) + 1
      FLAGS.iterations_per_loop = FLAGS.save_checkpoints_steps 
    # end
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(processor.label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  return estimator
 
class Predict_Parallel_Server():
    def __init__(self, child_pipe):
        #Measure performance
        self.t1 = time.time()
        self.child_pipe = child_pipe
        
    #---Server side---#
    def generate_from_pipe(self):
            """ Generator which yields items from the input pipe.
            """
            while True:
                feed = self.child_pipe.recv()
                if FLAGS.verbose:
                    tf.logging.info(' Yielding from Input Pipe')
                yield feed
                
    def pipe_based_predict_input_fn(self, params):
        if FLAGS.verbose:
            tf.logging.info(" Input Pipe is called")

        batch_size = params["batch_size"]

        # Fetch the inputs from the input pipe
        dataset = tf.data.Dataset.from_generator(self.generate_from_pipe,
                                                 output_types={'input_ids': tf.int32,
                                                               'input_mask': tf.int32,
                                                               'segment_ids': tf.int32,
                                                               'label_ids': tf.int32},
                                                 output_shapes={'input_ids': (FLAGS.max_seq_length),
                                                               'input_mask': (FLAGS.max_seq_length),
                                                               'segment_ids': (FLAGS.max_seq_length),
                                                               'label_ids': (1)})
                                                               
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=self.predict_drop_remainder)
        return dataset


    def predict_server_from_pipe(self):
        self.predict_drop_remainder = True if FLAGS.use_tpu else False
        
        estimator = build_bert_estimator()
        tf.logging.info(' Esitmator is created')
        
        #Measure performance
        self.t2 = time.time()
        startup_time = (self.t2 - self.t1)
        tf.logging.info(" Server is ready.")
        tf.logging.info(" Server Start-up time: %s seconds " % (str(startup_time)))

        tf.logging.info(' Start the loop to read pipe input')
        for prediction_result in estimator.predict(input_fn=self.pipe_based_predict_input_fn):
            if FLAGS.verbose:
                tf.logging.info(' Putting in output pipe')
                tf.logging.info(prediction_result)
            self.child_pipe.send(prediction_result)
            
def prepare_bert_server(name, child_pipe):
    global parallel_service
    
    parallel_service = Predict_Parallel_Server(child_pipe)
        
    #tf.logging.info('Predict process - sub pid: %d, ppid: %d' % (os.getpid(), os.getppid()))
    print(' Predict server process - sub pid: %d, ppid: %d' % (os.getpid(), os.getppid()))
    
    tf.app.run()
        
    #---End of Server side---#

class Predict_Parallel_Client():
    def __init__(self, pipe_parent):
        #Measure performance
        self.t1 = time.time()
        self.pipe_parent = pipe_parent
        self.prepare_bert_client()

    #---Client side---#
    def predict_request(self, input_data):
        predict_examples = self.processor.get_test_examples_from_list(input_data)
        return self.predict_processed_data(predict_examples)
        
    def predict_processed_data(self, predict_examples):
        num_actual_predict_examples = len(predict_examples)
        num_actual_sent_examples = num_actual_predict_examples

        # A fixed batch size for all batches, will help queue by batches
        # So we pad with fake examples which are ignored later on.
        while len(predict_examples) % FLAGS.predict_batch_size != 0:
            predict_examples.append(PaddingInputExample())
            num_actual_sent_examples += 1

        features = convert_examples_to_features(predict_examples, self.processor.label_list,
                                                FLAGS.max_seq_length, self.processor.tokenizer)

        if not FLAGS.silent:
            tf.logging.info(" ***** Running prediction*****")
            tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                            len(predict_examples), num_actual_predict_examples,
                            len(predict_examples) - num_actual_predict_examples)
            tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
            print(' Num of warming up data: '+str(len(features)))

        result = []
        
        #put number FLAGS.predict_batch_size of feature and get number FLAGS.predict_batch_size of result
        count = 0
        if FLAGS.parallel:
            for feature in features:
                if FLAGS.verbose:
                    tf.logging.info(' Putting into input pipe')
                self.pipe_parent.send({
                            "input_ids":
                                    feature.input_ids , 
                            "input_mask":
                                    feature.input_mask,
                            "segment_ids":
                                    feature.segment_ids,
                            "label_ids":
                                    [feature.label_id],
                        })
                count += 1
                if count == FLAGS.predict_batch_size:
                    if FLAGS.verbose:
                        tf.logging.info(" A batch completed. Start to get result.")
                    count = 0
                    for i in range(FLAGS.predict_batch_size):
                        r = self.pipe_parent.recv()
                        if FLAGS.verbose:
                            tf.logging.info(' Recieved from output pipe')
                            print(r)
                        result.append(r)  # The latest predictions generator
      
        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        output_predict_list = []
        num_written_lines = 0

        if not FLAGS.silent:
            tf.logging.info(" ***** Predict results *****")
        for (i, prediction) in enumerate(result):
            probabilities = prediction["probabilities"]
            if i >= num_actual_predict_examples:
                break
                
            output_line = "\t".join(
                str(class_probability)
                for class_probability in probabilities) 
            output_predict_list.append(output_line)
            
            num_written_lines += 1

        assert num_written_lines == num_actual_predict_examples

        with tf.gfile.GFile(output_predict_file, "a") as writer:
            if not FLAGS.silent:
                tf.logging.info(" ***** Write predict file *****")
            for output_line in output_predict_list:
                output_line = output_line + "\n"
                writer.write(output_line)
            writer.close()
        
        return output_predict_list

    def prepare_bert_client(self):
        tf.logging.info(' main - pid: %d, ppid: %d' % (os.getpid(), os.getppid()))
    
        # to be optimized for variable creation
        processors = {
            "sentij": SentiJProcessor
        }

        tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                      FLAGS.init_checkpoint)

        task_name = FLAGS.task_name.lower()

        if task_name not in processors:
            raise ValueError("Task not found: %s" % (task_name))

        self.processor = processors[task_name]()

        self.processor.label_list = self.processor.get_labels(FLAGS.data_dir)

        self.processor.tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        if FLAGS.do_predict:
            #warmup with file to inital model before ready to start client
            tf.logging.info(" warmup starts")
            predict_examples = self.processor.get_test_examples(FLAGS.data_dir, "warmup-predict.tsv")
            warmup_result = self.predict_processed_data(predict_examples)
            if FLAGS.verbose:
                tf.logging.info(" warmup result:")
                tf.logging.info(warmup_result)
            
            tf.logging.info(" Client process started.")

            #Measure performance
            self.t2 = time.time()
            startup_time = (self.t2 - self.t1)
            tf.logging.info(" Client is ready.")
            tf.logging.info(" Client Start-up time: %s seconds " % (str(startup_time)))

        
    def stop_bert_as_my_service(self):
        tf.logging.info(" Stop on request. Bye.")
    #---End of Client side---#
        
def main(_):
  global parallel_service
  
  tf.logging.set_verbosity(tf.logging.INFO)
  
  # add subversion to output_dir
  if not( FLAGS.subversion is None or FLAGS.subversion == ''):
    FLAGS.output_dir = FLAGS.output_dir + '-' + FLAGS.subversion
  #end
  
  if not ( ( not FLAGS.do_train ) and ( not FLAGS.do_eval ) and FLAGS.do_predict ):
    raise ValueError(
        "This service model is a predictor only, must have `do_predict' be True and 'do_train' be False.")

  if FLAGS.use_tpu:
    raise ValueError(
        "This service model must have `use_tpu' be False.")

  tf.logging.info(" Starting service\n")
  parallel_service.predict_server_from_pipe() # this is a endless loop

def Initialize():
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  #tf.app.run()

  if not FLAGS.parallel:
    raise ValueError(
        "This service model version must have 'parallel' be True ")
  
  #Define a duplex pipe for data to predict and result
  pipe_parent, pipe_child = multiprocessing.Pipe(duplex=True)
  print(" Pipe created.")
  
  if FLAGS.do_predict:
    if FLAGS.parallel:
      prediction_process = multiprocessing.Process(target=prepare_bert_server, args=('PredictServer', pipe_child))
      prediction_process.start()
      print(" Predict server process kicked off in separate process.")

      parallel_client = Predict_Parallel_Client(pipe_parent)
  print(' End of initialization')
  
  return parallel_client

#-- demo code --#
def predict_data_input_fn():
    #Use keyboard input as demo
    data = input(">>")
    return [data]

def demo_client(parallel_client):
    print(' Start of bert_client')

    try:
        while True:
            input_data = predict_data_input_fn()
            output_data = parallel_client.predict_request(input_data)
            JS = output_data[0].split('\t')
            if float(JS[0]) > float(JS[1]):
                print('Positive ' + JS[0])
            else:
                print('Negatvie ' + JS[1])
            
    except (KeyboardInterrupt, EOFError) :
        parallel_client.stop_bert_as_my_service()
        sys.exit()
#-- end of demo code --#
                
if __name__ == "__main__":
  parallel_client = Initialize()
  demo_client(parallel_client)
