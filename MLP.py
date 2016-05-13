import os
import sys
import gzip
import time
import cPickle
import numpy
import theano
import theano.tensor as T

class HiddenLayer(object) :
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.input = input
        if W is None :
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
            
        if b is None :
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b

        lin_output = T.dot(input,self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W, self.b]

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype = theano.config.floatX
                ),
            name='W',
            borrow=True
        )
        self.b = theano.shared(
            value = numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
                ),
            name='b',
            borrow=True
            )
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x,axis=1)
        self.params = [self.W,self.b]

    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
                )
        if y.dtype.startswith('int') :
            return T.mean(T.neq(self.y_pred, y))
        else :
            raise NotImplementedError()
            

class MLP(object) :
    def __init__(self,rng,input,n_in,n_hidden,n_out):
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
            )
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
            )

        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            +abs(self.logRegressionLayer.W).sum()
            )
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            +(self.logRegressionLayer.W ** 2).sum()
            )
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
            )
        self.errors = self.logRegressionLayer.errors
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

def test_mlp(learning_rate=0.01, L1_reg=0.00,L2_reg=0.0001,n_epochs=10,
             dataset='mnist.pkl.gz',batch_size=20,n_hidden=500):
    datasets = load_data(dataset)
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size  
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size  
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size
    print '...buidling the model'
    
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    rng = numpy.random.RandomState(1234)

    classifier = MLP(
        rng=rng,
        input=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
        )

    cost = (
        classifier.negative_log_likelihood(y)
        +L1_reg *classifier.L1
        +L2_reg *classifier.L2_sqr
        )
    
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens = {
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )
    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens = {
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            }
        )
    gparams = [T.grad(cost, param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
        ]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
    
    print '...training'

    patience = 10000
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience/2)

    best_validation_loss = numpy.inf
    best_iter=0
    test_score=0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch+1
        for minibatch_index in xrange(n_train_batches) :
            minibatch_avg_cost = train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index
            if (iter + 1)%validation_frequency == 0:
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print (
                    'epoch %i, minibathc %i%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                        )
                    )
                if this_validation_loss < best_validation_loss :
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                        ):
                        patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        test_losses = [test_model(i) for i
                                       in xrange(n_test_batches)]
                        test_scores = numpy.mean(test_losses)
                        print (('   epoch %i, minibatch %i%i, text error of '
                                'best model %f %%')%
                               (epoch, minibatch_index + 1, n_train_batches,
                                test_score * 100.))
                        if patience <= iter :
                            done_looping = True
                            break
                end_time = time.clock
                print (('Optimization complete. Best validation score of %f %% '
                        'obtained at iteration %i, with test performance %f %%')%
                       (best_validation_loss * 100., best_iter + 1, test_score *100.))
                print >> sys.stderr, ('The code for file ' +
                                      os.path.split(__file__)[1] +
                                      ' ran for %.2fm' %((end_time - start_time) / 60.))
                

def load_data(dataset):
    dataset = r'mnist.pkl.gz'
    
#从"mnist.pkl.gz"里加载train_set, valid_set, test_set，它们都是包括label的
#主要用到python里的gzip.open()函数,以及 cPickle.load()。
#‘rb’表示以二进制可读的方式打开文件
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
   

#将数据设置成shared variables，主要时为了GPU加速，只有shared variables才能存到GPU memory中
#GPU里数据类型只能是float。而data_y是类别，所以最后又转换为int返回
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')


    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
        
if __name__ == '__main__' :
    test_mlp()
    
        
