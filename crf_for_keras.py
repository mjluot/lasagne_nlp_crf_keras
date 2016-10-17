import lasagne_nlp.utils.data_processor as data_processor
import argparse
import numpy as np
from os import path
from output import prediction_directory
from settings import fill_args
#from lasagne_nlp.utils.objectives import crf_loss, crf_accuracy

import theano.tensor as T
import theano

from lasagne import init
from lasagne.layers import MergeLayer
from keras.engine.topology import Layer


def theano_logsumexp(x, axis=None):
    """
    Compute log(sum(exp(x), axis=axis) in a numerically stable
    fashion.
    Parameters
    ----------
    x : tensor_like
        A Theano tensor (any dimension will do).
    axis : int or symbolic integer scalar, or None
        Axis over which to perform the summation. `None`, the
        default, performs over all axes.
    Returns
    -------
    result : ndarray or scalar
        The result of the log(sum(exp(...))) operation.
    """

    xmax = x.max(axis=axis, keepdims=True)
    xmax_ = x.max(axis=axis)
    return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))


def crf_loss(targets, energies):
    """
    compute minus log likelihood of crf as crf loss.
    :param energies: Theano 4D tensor
        energies of each step. the shape is [batch_size, n_time_steps, num_labels, num_labels],
        where the pad label index is at last.
    :param targets: Theano 2D tensor
        targets in the shape [batch_size, n_time_steps]
    :param masks: Theano 2D tensor
        masks in the shape [batch_size, n_time_steps]
    :return: Theano 1D tensor
        an expression for minus log likelihood loss.
    """

    #targets = T.cast(targets, 'int32')
    #print targets.eval().shape
    #We are so sorry ;_;
    #targets = targets[-1][-1]
    #print energies.eval().shape


    assert energies.ndim == 4
    assert targets.ndim == 2
    #assert masks.ndim == 2

    def inner_function(energies_one_step, targets_one_step, prior_partition, prev_label, tg_energy):
        """

        :param energies_one_step: [batch_size, t, t]
        :param targets_one_step: [batch_size]
        :param prior_partition: [batch_size, t]
        :param prev_label: [batch_size]
        :param tg_energy: [batch_size]
        :return:
        """

        partition_shuffled = prior_partition.dimshuffle(0, 1, 'x')
        partition_t = theano_logsumexp(energies_one_step + partition_shuffled, axis=1)

        
        print [T.arange(energies_one_step.shape[0]), prev_label, targets_one_step]
        return [partition_t, targets_one_step,
                tg_energy + energies_one_step[T.cast(T.arange(energies_one_step.shape[0]),'int32'), prev_label, targets_one_step]]

    # Input should be provided as (n_batch, n_time_steps, num_labels, num_labels)
    # but scan requires the iterable dimension to be first
    # So, we need to dimshuffle to (n_time_steps, n_batch, num_labels, num_labels)
    energies_shuffled = energies.dimshuffle(1, 0, 2, 3)
    targets_shuffled = targets.dimshuffle(1, 0)
    #masks_shuffled = masks.dimshuffle(1, 0)

    # initials should be energies_shuffles[0, :, -1, :]
    init_label = T.cast(T.fill(energies[:, 0, 0, 0], -1), 'int32')
    energy_time0 = energies_shuffled[0]
    target_time0 = targets_shuffled[0]

    #print dir(targets_shuffled)


    #print T.arange(energy_time0.shape[0]), init_label, target_time0


    initials = [energies_shuffled[0, :, -1, :], target_time0,
                energy_time0[T.arange(energy_time0.shape[0]), init_label, target_time0]]

    #print 'Before scan'
    #print 'initials', initials
    #initials = [T.cast(X, 'int32') for X in initials]
    #print initials
    #energies_shuffled = T.cast(energies_shuffled, 'int32')
    targets_shuffled = T.cast(targets_shuffled, 'int32')

    [partitions, _, target_energies], _ = theano.scan(fn=inner_function, outputs_info=initials,
                                                      sequences=[energies_shuffled[1:], targets_shuffled[1:],
                                                                ])
    partition = partitions[-1]
    target_energy = target_energies[-1]
    loss = theano_logsumexp(partition, axis=1) - target_energy
    return loss


class CRFLayer(Layer):
    """
    lasagne_nlp.networks.crf.CRFLayer(incoming, num_labels,
    mask_input=None, W=init.GlorotUniform(), b=init.Constant(0.), **kwargs)

    Parameters
    ----------
    incoming : a :class:`lasagne.layers.Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape.
        The output of this layer should be a 3D tensor with shape
        ``(batch_size, input_length, num_input_features)``
    num_labels : int
        The number of labels of the crf layer
    mask_input : :class:`lasagne.layers.Layer`
        Layer which allows for a sequence mask to be input, for when sequences
        are of variable length.  Default `None`, which means no mask will be
        supplied (i.e. all sequences are of the same length).
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a tensor with shape ``(num_inputs, num_units)``,
        where ``num_inputs`` is the size of the second dimension of the input.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_units,)
    """

    #Remove incoming
    def __init__(self, num_labels, **kwargs):
        # This layer inherits from a MergeLayer, because it can have two
        # inputs - the layer input, and the mask.
        # We will just provide the layer input as incomings, unless a mask input was provided.

        #self.input_shape = incoming.output_shape
        #incomings = [incoming]
        #self.mask_incoming_index = -1
        #if mask_input is not None:
        #    incomings.append(mask_input)
        #    self.mask_incoming_index = 1

        super(CRFLayer, self).__init__(**kwargs)
        self.num_labels = num_labels + 1
        self.pad_label_index = num_labels

        #if b is None:
        #    self.b = None
        #else:
        #    self.b = self.add_param(b, (self.num_labels, self.num_labels), name="b", regularizable=False)

    def build(self, input_shape):

        #print input_shape
        num_inputs = input_shape[2]
        #print num_inputs
        #print input_shape
        #self.input_shape = input_shape

        rng = np.random.RandomState(1337)
        W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (num_inputs + self.num_labels + self.num_labels)),
                    high=np.sqrt(6. / (num_inputs + self.num_labels + self.num_labels)),
                    size=(num_inputs, self.num_labels, self.num_labels)
                ),
                dtype=np.float32
            )

        self.W = theano.shared(value=W_values, name='W', borrow=True)#self.add_param(W, (num_inputs, self.num_labels, self.num_labels), name="W")
        self.trainable_weights = [self.W]

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], self.num_labels, self.num_labels

    #mask is out for now
    def call(self, input, mask=None):
        """
        Compute this layer's output function given a symbolic input variable.

        Parameters
        ----------
        :param inputs: list of theano.TensorType
            `inputs[0]` should always be the symbolic input variable.  When
            this layer has a mask input (i.e. was instantiated with
            `mask_input != None`, indicating that the lengths of sequences in
            each batch vary), `inputs` should have length 2, where `inputs[1]`
            is the `mask`.  The `mask` should be supplied as a Theano variable
            denoting whether each time step in each sequence in the batch is
            part of the sequence or not.  `mask` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``mask[i, j] = 1`` when ``j <=
            (length of sequence i)`` and ``mask[i, j] = 0`` when ``j > (length
            of sequence i)``.
        :return: theano.TensorType
            Symbolic output variable.
        """
        #input = inputs[0]
        #mask = None
        #if self.mask_incoming_index > 0:
        #    mask = inputs[self.mask_incoming_index]

        # compute out by tensor dot ([batch, length, input] * [input, num_label, num_label]
        # the shape of out should be [batch, length, num_label, num_label]
        out = T.tensordot(input, self.W, axes=[[2], [0]])

        #Bias is gone!
        #if self.b is not None:
        #    b_shuffled = self.b.dimshuffle('x', 'x', 0, 1)
        #    out = out + b_shuffled

        #if mask is not None:
        #    mask_shuffled = mask.dimshuffle(0, 1, 'x', 'x')
        #    out = out * mask_shuffled

        return out


def main():

    '''
    parser = argparse.ArgumentParser(description='Tuning with bi-directional LSTM-CNN-CRF')
    parser.add_argument('--embedding', choices=['word2vec', 'glove', 'senna', 'random'], help='Embedding for words',
                        required=True)
    parser.add_argument('--embedding_dict', default=None, help='path for embedding dict')
    parser.add_argument('--oov', choices=['random', 'embedding'], help='Embedding for oov word', required=True)
    parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    parser.add_argument('--datadir')
    args = parser.parse_args()
    '''

    oov = 'embedding' #args.oov
    embedding = 'word2vec'#args.embedding
    embedding_path = '/home/mjluot/data/vectors/wikipedia-pubmed-and-PMC-w2v.bin'#args.embedding_dict
    train_path = '/home/mjluot/data/tagging-corpora/ner/anatem/train.tsv'#args.train
    dev_path = '/home/mjluot/data/tagging-corpora/ner/anatem/devel.tsv'#args.dev
    test_path = '/home/mjluot/data/tagging-corpora/ner/anatem/test.tsv'#args.test
    fine_tune=False

    X_train, Y_train, mask_train, X_dev, Y_dev, mask_dev, X_test, Y_test, mask_test, \
    embedd_table, label_alphabet, \
    C_train, C_dev, C_test, char_embedd_table = data_processor.load_dataset_sequence_labeling(train_path, dev_path,
                                                                                              test_path, oov=oov,
                                                                                              fine_tune=fine_tune,
                                                                                              embedding=embedding,
                                                                                              embedding_path=embedding_path,
                                                                                              use_character=True)

    #Y_train = Y_train.reshape((1,1,Y_train.shape[-2], Y_train.shape[-1]))
    num_labels = label_alphabet.size() - 1

    #Okay, now I've got some training data!
    #import pdb;pdb.set_trace()

    from keras.models import Sequential, Graph
    from training import Model
    from keras.layers import Dense, Dropout, Activation, Merge, Input, merge
    from keras.optimizers import SGD
    from keras import backend as K


    t_input = Input(shape=(X_train.shape[1], X_train.shape[2]), name='t_input', dtype='float32') 
    crf = CRFLayer(200)
    
    out = crf(t_input)
    model = Model(input=t_input, output=out)
    model.compile(optimizer='sgd', loss=crf_loss, target_placeholder=[K.placeholder(ndim=2, dtype='int32', name='out_target')])

    import pdb;pdb.set_trace()



if __name__=='__main__':
    main()
