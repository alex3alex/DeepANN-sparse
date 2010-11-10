try:
    from SimpledAclass import *
except ImportError:
    from deepANN.SimpledAclass import *
import cPickle
import os
import os.path
import time
import sys
import math

from jobman.tools import DD,expand
from jobman.parse import filemerge

from common.stats import stats
from common.str import percent
from common.movingaverage import MovingAverage
from common.file import myopen

# TRAINFUNC is a handle to the model's training function. It is a global
# because it is connected to internal state in the Model. Each time the
# model changes, update TRAINFUNC!
TRAINFUNC       = None

globalstate = None

def rebuildunsup(model,LR,NOISE_LVL,ACTIVATION_REGULARIZATION_COEFF, WEIGHT_REGULARIZATION_COEFF, batchsize,train):
    """
    Modify the global TRAINFUNC and TESTFUNC.
    TODO: FIXME! Is it possible not to use global state? If the TRAINFUNC
    and TESTFUNC are connected to model state, then it is unavoidable that
    TRAINFUNC and TESTFUNC should be treated as things with side-effects.
    """

    global TRAINFUNC
#    givens = {}
#    index = T.lscalar()
#    givens.update({model.x : train[index*batchsize:(index+1)*batchsize]})
    # TODO: Remove params above that are not necessary: batchsize, train, LR
    x = T.dmatrix()
    params = [T.dmatrix(), T.dmatrix(), T.dvector(), T.dvector()]
    # TODO: Remove learning_rate below
    (cost,update) = model.get_cost_updates(x, params[0], params[1], params[2], params[3], corruption_level = NOISE_LVL, learning_rate = LR, l1reg = ACTIVATION_REGULARIZATION_COEFF, l2reg = WEIGHT_REGULARIZATION_COEFF)
    TRAINFUNC = theano.function([x] + params, [cost] + update)
    # TODO: Now do the update

def createlibsvmfile(model,datafiles,dataout):
    print >> sys.stderr, 'Creating libsvm file %s (model=%s, datafiles=%s)...' % (repr(dataout), repr(model),datafiles)
    print >> sys.stderr, stats()

    x = T.dmatrix()
    params = [T.dmatrix(), T.dmatrix(), T.dvector(), T.dvector()]
    model.x = x
    model.W, model.W_prime, model.b, model.b_prime = params
    model.params = [model.W, model.W_prime, model.b, model.b_prime]

    outputs = [model.get_hidden_values(model.x)]
    func = theano.function([model.x] + params,outputs)

#    print >> sys.stderr, 'REMOVEME: about to read'
#    print >> sys.stderr, stats()

    f = myopen(datafiles[0],'r')
    instances = numpy.asarray(cPickle.load(f),dtype=theano.config.floatX)
    f.close()
    f = myopen(datafiles[1],'r')
    labels = numpy.asarray(cPickle.load(f),dtype = 'int64')
    f.close()
    f = myopen(dataout,'w')

#    print >> sys.stderr, 'REMOVEME: about to iterate'
#    print >> sys.stderr, stats()

#    params = [model.Wvalue, model.W_primevalue, model.bvalue, model.b_primevalue]
    for i in range(globalstate.NB_MAX_TRAINING_EXAMPLES_SVM/globalstate.BATCH_CREATION_LIBSVM):
#        print >> sys.stderr, 'REMOVEME: about to do %d' % i
#        print >> sys.stderr, stats()
        textr = ''

        assert globalstate.BATCH_CREATION_LIBSVM == 1       # Don't want to select indices from more than one example
        x = instances[globalstate.BATCH_CREATION_LIBSVM*i:globalstate.BATCH_CREATION_LIBSVM*(i+1),:]
        nonzeros = frozenset(x.nonzero()[1])
#        print >> sys.stderr, nonzeros
#        print >> sys.stderr, len(nonzeros)

        indices = list(nonzeros)
#        # TODO: Don't duplicate this code, which also appears about one hundred lines down.
#        x = x[:,indices]
#        params = [model.Wvalue[indices], model.W_primevalue[:,indices], model.bvalue, model.b_primevalue[indices]]
#        rep = func(x, *params)[0]

        rep = func(x[:,indices], model.Wvalue[indices], model.W_primevalue[:,indices], model.bvalue, model.b_primevalue[indices])[0]

        for l in range(rep.shape[0]):
            textr += '%s '%labels[globalstate.BATCH_CREATION_LIBSVM*i+l]
            idx = rep[l,:].nonzero()[0]
            for j,v in zip(idx,rep[l,idx]):
                textr += '%s:%s '%(j,v)
            textr += '\n'
        f.write(textr)
    del instances,labels
    f.close()
    print >> sys.stderr, "...done creating libsvm files"
    print >> sys.stderr, stats()

def svm_validation_for_one_trainsize_and_one_C(C, nbinputs,numruns,datatrainsave,datatestsave,PATH_SAVE):
    """
    Train an SVM using some C on nbinputs training examples, for numrums runs.
    Return:
        testerr,testerrdev,trainerr,trainerrdev
    """
    print >> sys.stderr, "\t\tTraining SVM with C=%f, nbinputs=%d, numruns=%d" % (C, nbinputs,numruns)

    os.system('%s -s 4 -c %s -l %s -r %s -q %s %s %s > /dev/null 2> /dev/null'%(globalstate.SVMRUNALL_PATH,C,nbinputs,numruns,datatrainsave,datatestsave,PATH_SAVE+'/currentsvm.txt'))
    results = myopen(PATH_SAVE+'/currentsvm.txt','r').readline()[:-1].split(' ')
    os.remove(PATH_SAVE+'/currentsvm.txt')
    trainerr       = float(results[1])
    trainerrdev    = float(results[2])
    testerr        = float(results[3])
    testerrdev     = float(results[4])
    return testerr,testerrdev,trainerr,trainerrdev


def svm_validation_for_one_trainsize(nbinputs,numruns,datatrainsave,datatestsave,PATH_SAVE):
    """
    Train an SVM on nbinputs training examples, for numrums runs.
    Choose the value of C using a linesearch to minimize the testerr.
    Return:
        C,testerr,testerrdev,trainerr,trainerrdev

    MAXSTEPS is the number of steps performed in the line search.
    STEPFACTOR is the initial step size.
    """
    MAXSTEPS=globalstate.SVM_MAXSTEPS
    STEPFACTOR=globalstate.SVM_STEPFACTOR
    INITIALC=globalstate.SVM_INITIALC

    print >> sys.stderr, 'Starting SVM validation for %s examples (numrums=%d, datatrainsave=%s, datatestsave=%s, PATH_SAVE=%s, MAXSTEPS=%d, STEPFACTOR=%f, INITIALC=%f)...' % (nbinputs, numruns,datatrainsave,datatestsave, PATH_SAVE,MAXSTEPS, STEPFACTOR, INITIALC)
    print >> sys.stderr, stats()

    Ccurrent = INITIALC
    Cstepfactor = STEPFACTOR
    Cnew = Ccurrent * Cstepfactor

    C_to_allstats = {}
    Cbest = None

    while len(C_to_allstats) < MAXSTEPS:
        if Ccurrent not in C_to_allstats:
            # Compute the validation statistics for the current C
            testerr,testerrdev,trainerr,trainerrdev = svm_validation_for_one_trainsize_and_one_C(Ccurrent, nbinputs,numruns,datatrainsave,datatestsave,PATH_SAVE)
            C_to_allstats[Ccurrent] = (testerr,testerrdev,trainerr,trainerrdev)
        if Cnew not in C_to_allstats:
            # Compute the validation statistics for the next C
            testerr,testerrdev,trainerr,trainerrdev = svm_validation_for_one_trainsize_and_one_C(Cnew, nbinputs,numruns,datatrainsave,datatestsave,PATH_SAVE)
            C_to_allstats[Cnew] = (testerr,testerrdev,trainerr,trainerrdev)
        # If Cnew has a lower test err than Ccurrent, then continue stepping in this direction
        if C_to_allstats[Cnew][0] < C_to_allstats[Ccurrent][0]:
            print >> sys.stderr, "\ttesterr[Cnew %f] = %f < testerr[Ccurrent %f] = %f" % (Cnew, C_to_allstats[Cnew][0], Ccurrent, C_to_allstats[Ccurrent][0])
            if Cbest is None or C_to_allstats[Cnew][0] < C_to_allstats[Cbest][0]:
                Cbest = Cnew
                print >> sys.stderr, "\tNEW BEST: Cbest <= %f, testerr[Cbest] = %f" % (Cbest, C_to_allstats[Cbest][0])
            Ccurrent = Cnew
            Cnew *= Cstepfactor
            print >> sys.stderr, "\tPROCEED: Cstepfactor remains %f, Ccurrent is now %f, Cnew is now %f" % (Cstepfactor, Ccurrent, Cnew)
        # Else, reverse the direction and reduce the step size by sqrt.
        else:
            print >> sys.stderr, "\ttesterr[Cnew %f] = %f > testerr[Ccurrent %f] = %f" % (Cnew, C_to_allstats[Cnew][0], Ccurrent, C_to_allstats[Ccurrent][0])
            if Cbest is None or C_to_allstats[Ccurrent][0] < C_to_allstats[Cbest][0]:
                Cbest = Ccurrent
                print >> sys.stderr, "\tCbest <= %f, testerr[Cbest] = %f" % (Cbest, C_to_allstats[Cbest][0])
            Cstepfactor = 1. / math.sqrt(Cstepfactor)
            Cnew = Ccurrent * Cstepfactor
            print >> sys.stderr, "\tREVERSE: Cstepfactor is now %f, Ccurrent remains %f, Cnew is now %f" % (Cstepfactor, Ccurrent, Cnew)

    allC = C_to_allstats.keys()
    allC.sort()
    for C in allC:
        print >> sys.stderr, "\ttesterr[C %f] = %f" % (C, C_to_allstats[C][0]),
        if C == Cbest: print >> sys.stderr, " *best* (testerr = %f, testerrdev = %f, trainerr = %f, trainerrdev = %f)"% C_to_allstats[C]
        else: print >> sys.stderr, ""
    print >> sys.stderr, '...done with SVM validation for %s examples (numrums=%d, datatrainsave=%s, datatestsave=%s)' % (nbinputs, numruns,datatrainsave,datatestsave)
    print >> sys.stderr, stats()

    return [Cbest] + list(C_to_allstats[Cbest])

def svm_validation(err, epoch, model, train,datatrain,datatrainsave,datatest,datatestsave, VALIDATION_TRAININGSIZE, VALIDATION_RUNS_FOR_EACH_TRAININGSIZE, PATH_SAVE, PATH_DATA, NAME_DATATEST):
    """
    Perform full SVM validation.
    """
    print >> sys.stderr, "Validating (err=%s,epoch=%s,model=%s,train=%s,datatrain=%s,datatrainsave=%s,datatest=%s,datatestsave=%s,VALIDATION_TRAININGSIZE=%s,VALIDATION_RUNS_FOR_EACH_TRAININGSIZE=%s,PATH_SAVE=%s)..." % (err, epoch, model,train,datatrain,datatrainsave,datatest,datatestsave, VALIDATION_TRAININGSIZE, VALIDATION_RUNS_FOR_EACH_TRAININGSIZE, PATH_SAVE)
    print >> sys.stderr, stats()

    createlibsvmfile(model,datatrain,datatrainsave)
    createlibsvmfile(model,datatest,datatestsave)

    for trainsize in VALIDATION_TRAININGSIZE:
        print trainsize
        print VALIDATION_RUNS_FOR_EACH_TRAININGSIZE
        C,testerr,testerrdev,trainerr,trainerrdev = svm_validation_for_one_trainsize(trainsize,VALIDATION_RUNS_FOR_EACH_TRAININGSIZE[`trainsize`],datatrainsave,datatestsave,PATH_SAVE)
        err[trainsize].update({epoch:(C,testerr,testerrdev,trainerr,trainerrdev)})

    for trainsize in VALIDATION_TRAININGSIZE:
        print >> sys.stderr, 'VALIDATION: epoch %d / trainsize %d / svm error' % ( epoch, trainsize) ,err[trainsize][epoch]
    print >> sys.stderr, stats()

    if epoch != 0:
        f = myopen('err.pkl','w')
        for trainsize in VALIDATION_TRAININGSIZE:
            cPickle.dump(err[trainsize],f,-1)
        f.close()
    print >> sys.stderr, "...done validating (err=%s,epoch=%s,model=%s,train=%s,datatrain=%s,datatrainsave=%s,datatest=%s,datatestsave=%s,VALIDATION_TRAININGSIZE=%s,VALIDATION_RUNS_FOR_EACH_TRAININGSIZE=%s,PATH_SAVE=%s)" % (err, epoch, model,train,datatrain,datatrainsave,datatest,datatestsave, VALIDATION_TRAININGSIZE, VALIDATION_RUNS_FOR_EACH_TRAININGSIZE, PATH_SAVE)
    print >> sys.stderr, stats()

#def runtrainfunc(TRAINFUNC, x, params):
def runtrainfunc(TRAINFUNC, x, model, indices):
    print len(indices)
#    r = TRAINFUNC(x, *params)
    # FIXME
    r = TRAINFUNC(x[:,indices], model.Wvalue[indices], model.W_primevalue[:,indices], model.bvalue, model.b_primevalue[indices])
#    r = TRAINFUNC(x, model.Wvalue, model.W_primevalue, model.bvalue, model.b_primevalue)
    assert len(r) == 5
    reconstruction_error_over_batch = r[0]
    gparams = r[1:]
    return reconstruction_error_over_batch, gparams

def get_indices(x):
    assert x.shape[0] == 1      # Assume minibatch

    nonzeros = x.nonzero()[1]

    # It is conceivable that some of these zeros overlap with each other or other nonzeros
    # TODO: Make this value a hyperparameters
#    ZEROS = 20
    ZEROS = 100-len(frozenset(nonzeros))
    # TODO: Seed RNG with hyperparam seed
    import random
#    print x.shape
    zeros = [random.randint(0, x.shape[1]-1) for i in range(ZEROS)]
#    print zeros, nonzeros
    indices = list(frozenset(nonzeros) | frozenset(zeros))
    indices.sort()
#    print len(indices)
    return indices

#def slice_x_and_params(x, model, indices):
##    print len(indices), indices
##    print x.shape
#    x = x[:,indices]
##    print x.shape
##    params = [model.Wvalue, model.W_primevalue, model.bvalue, model.b_primevalue]
##    print [p.shape for p in params]
#    params = [model.Wvalue[indices], model.W_primevalue[:,indices], model.bvalue, model.b_primevalue[indices]]
##    print [p.shape for p in params]
#    return params, x

#def apply_updates(params, gparams):
#    for param, gparam in zip(params, gparams):
#        param += gparam

#def apply_updates(model, indices, gparams):
#    model.Wvalue[indices]           += gparams[0]
#    model.W_primevalue[:,indices]   += gparams[1]
#    model.bvalue                    += gparams[2]
#    model.b_primevalue[indices]     += gparams[3]
def apply_updates(model, indices, gparams):
#    print [model.Wvalue[indices].shape, model.W_primevalue[:,indices].shape, model.bvalue.shape, model.b_primevalue[indices].shape]
#    print [n.shape for n in gparams]
    apply_updates1(model, indices, gparams)
    apply_updates2(model, indices, gparams)
    apply_updates3(model, indices, gparams)
    apply_updates4(model, indices, gparams)
def apply_updates1(model, indices, gparams):
    model.Wvalue[indices]           += gparams[0]
def apply_updates2(model, indices, gparams):
#    print gparams[1]
#    print model.W_primevalue[:,indices]
    model.W_primevalue[:,indices]   += gparams[1]
#    print model.W_primevalue[:,indices]
def apply_updates3(model, indices, gparams):
    model.bvalue                    += gparams[2]
def apply_updates4(model, indices, gparams):
    model.b_primevalue[indices]     += gparams[3]

def training_step(BATCHSIZE, j, train, TRAINFUNC, model, train_reconstruction_error_mvgavg):
#    print "REMOVEME running TRAINFUNC"
    assert BATCHSIZE == 1       # This index sampling training technique might not make sense with BATCHSIZE > 1

    x = train.container.value[j*BATCHSIZE:(j+1)*BATCHSIZE]
    indices = get_indices(x)
#    params, x = slice_x_and_params(x, model, indices)

    reconstruction_error_over_batch, gparams = runtrainfunc(TRAINFUNC, x, model, indices)
    train_reconstruction_error_mvgavg.add(reconstruction_error_over_batch)
#    print reconstruction_error_over_batch
    apply_updates(model, indices, gparams)


def NLPSDAE_help(state,channel):
    """ In this simplified version we only train the first layer, the hyperparameters could not be lists"""

    global globalstate
    globalstate = state
    globalstate.SVMRUNALL_PATH = os.path.join(globalstate.SVMPATH, "run_all")

    print >> sys.stderr, globalstate.SVMRUNALL_PATH
    assert os.access(globalstate.SVMRUNALL_PATH, os.X_OK)

    # Hyper-parameters
    LR = state.lr#list
    ACT = state.act #list
    DEPTH = state.depth
    assert DEPTH == 1 #simplified code only for the first layer
    N_HID = state.n_hid #list
    NOISE = state.noise #list
    NOISE_LVL = state.noise_lvl#list
    ACTIVATION_REGULARIZATION_TYPE = state.activation_regularization_type
    ACTIVATION_REGULARIZATION_COEFF = state.activation_regularization_coeff #list
    WEIGHT_REGULARIZATION_TYPE = state.weight_regularization_type
    WEIGHT_REGULARIZATION_COEFF = state.weight_regularization_coeff #list
    NEPOCHS = state.nepochs #list
    VALIDATION_RUNS_FOR_EACH_TRAININGSIZE = state.validation_runs_for_each_trainingsize #dict from trainsize string to number of validation runs at this training size
    VALIDATION_TRAININGSIZE = [int(trainsize) for trainsize in VALIDATION_RUNS_FOR_EACH_TRAININGSIZE] # list
    VALIDATION_TRAININGSIZE.sort()
    EPOCHSTEST = state.epochstest #list
    BATCHSIZE = state.batchsize
    PATH_SAVE = channel.remote_path if hasattr(channel,'remote_path') else channel.path
    NB_FILES = state.nb_files
    PATH_DATA = state.path_data
    NAME_DATA = state.name_traindata
    NAME_LABEL = state.name_trainlabel
    NAME_DATATEST = state.name_testdata
    NAME_LABELTEST = state.name_testlabel
    MODEL_RELOAD = state.model_reload if hasattr(state,'model_reload') else None
    NINPUTS = state.ninputs          # Number of input dimensions
    INPUTTYPE = state.inputtype
    RULE = state.rule if hasattr(state,'rule') else None
    RandomStreams(state.seed)
    numpy.random.seed(state.seed)
    datatrain = (PATH_DATA+NAME_DATA+'_1.pkl.gz',PATH_DATA+NAME_LABEL+'_1.pkl.gz')
    datatrainsave = PATH_SAVE+'/train.libsvm'
    datatest = (PATH_DATA+NAME_DATATEST+'_1.pkl.gz',PATH_DATA+NAME_LABELTEST+'_1.pkl.gz')
    datatestsave = PATH_SAVE+'/test.libsvm'

    depthbegin = 0

    #monitor best performance for reconstruction and classification
    state.besterr = dict([(`trainsize`, []) for trainsize in VALIDATION_TRAININGSIZE])
    state.besterrepoch = dict([(`trainsize`, []) for trainsize in VALIDATION_TRAININGSIZE])

    filename = PATH_DATA + NAME_DATATEST + '_1.pkl.gz'
    print filename
    f =myopen(filename,'r')
    train = theano.shared(numpy.asarray(cPickle.load(f),dtype=theano.config.floatX))
    f.close()
    normalshape = train.value.shape
    
    model=dA(numpy.random,RandomStreams(),input = None, n_visible = NINPUTS, n_hidden = N_HID, act = ACT, noise = NOISE)

    #RELOAD previous model
    channel.save()

    err = dict([(trainsize, {}) for trainsize in VALIDATION_TRAININGSIZE])
    rebuildunsup(model,LR,NOISE_LVL,ACTIVATION_REGULARIZATION_COEFF, WEIGHT_REGULARIZATION_COEFF, BATCHSIZE,train)

    epoch = 0
    if epoch in EPOCHSTEST:
        svm_validation(err, epoch, model,train,datatrain,datatrainsave,datatest,datatestsave, VALIDATION_TRAININGSIZE, VALIDATION_RUNS_FOR_EACH_TRAININGSIZE, PATH_SAVE, PATH_DATA, NAME_DATATEST)
        channel.save()

    train_reconstruction_error_mvgavg = MovingAverage()
    for epoch in xrange(1,NEPOCHS+1):
        time1 = time.time()
        state.currentepoch = epoch
        for filenb in xrange(1,NB_FILES + 1):
            print >> sys.stderr, "\t\tAbout to read file %s..." % percent(filenb, NB_FILES)
            print >> sys.stderr, "\t\t", stats()
#                initial_file_time = time.time()
            f =myopen(PATH_DATA + NAME_DATA +'_%s.pkl.gz'%filenb,'r')
            object = numpy.asarray(cPickle.load(f),dtype=theano.config.floatX)
            print >> sys.stderr, "\t\t...read file %s" % percent(filenb, NB_FILES)
            print >> sys.stderr, "\t\t", stats()
            # The last training file is not of the same shape as the other training files.
            # So, to avoid a GPU memory error, we want to make sure it is the same size.
            # In which case, we pad the matrix but keep track of how many n (instances) there actually are.
            # TODO: Also want to pad trainl
            if object.shape == normalshape:
                train.container.value[:] = object
                currentn = normalshape[0]
                del object
            else:
                train.container.value[:] = numpy.concatenate([object,\
                    numpy.zeros((normalshape[0]-object.shape[0],normalshape[1]),dtype=theano.config.floatX)])
                currentn = object.shape[0]
                del object
            f.close()
            if train.value.min() < 0:
                print >> sys.stderr, "WARNING: Negative input, currently input should be positive"
            if train.value.max() > 1. and INPUTTYPE!='tfidf':
                print >> sys.stderr, "WARNING: Some inputs are > 1, without tfidf inputtype, it should be in the range [0,1]" 
            for j in range(currentn/BATCHSIZE):
                training_step(BATCHSIZE, j, train, TRAINFUNC, model, train_reconstruction_error_mvgavg)

                # REMOVEME
                if j > 100: sys.exit(0)

            print >> sys.stderr, "\t\tAt epoch %d, finished training over file %s, online reconstruction error %s" % (epoch, percent(filenb, NB_FILES),train_reconstruction_error_mvgavg)
            print >> sys.stderr, "\t\t", stats()
        print >> sys.stderr, '...finished training epoch #%s' % percent(epoch,NEPOCHS)
        print >> sys.stderr, stats()
#           sys.stderr.flush()
#           or maybe you need
        #jobman cachesync
        
        if epoch in EPOCHSTEST:
            svm_validation(err, epoch, model,train,datatrain,datatrainsave,datatest,datatestsave, VALIDATION_TRAININGSIZE, VALIDATION_RUNS_FOR_EACH_TRAININGSIZE, PATH_SAVE, PATH_DATA, NAME_DATATEST)

        channel.save()
        if len(EPOCHSTEST)!=0:
            for trainsize in VALIDATION_TRAININGSIZE:
                errvector = err[trainsize].values()
                for k in range(len(errvector)):
                    errvector[k] = errvector[k][1]
                errmin = numpy.min(errvector)
                for k in err[trainsize].keys():
                    if err[trainsize][k][1] == errmin:
                        state.besterr[`trainsize`] += [err[trainsize][k]]
                        state.besterrepoch[`trainsize`] += [k]
        else:
            for trainsize in VALIDATION_TRAININGSIZE:
                state.besterr[`trainsize`] += [None]
                state.besterrepoch[`trainsize`] += [None]
        print >> sys.stderr, stats()
    return channel.COMPLETE


def NLPSDAE(state,channel):
#    import cProfile
#    import DARPAscript_simplified
#    cProfile.run('DARPAscript_simplified.NLPSDAE_help(state,channel)', 'DARPAscript_simplified.stats')
#    cProfile.run('NLPSDAE_help(state,channel)', 'DARPAscript_simplified.stats')
    NLPSDAE_help(state,channel)
