# Run me like this:
# sqsub -q serial -r 2d -n 1 --memperproc=3.5G -o batch-bestrectifier_simplified.out ./batch-bestrectifier_simplified.sh

#PYTHONPATH=$PYTHONPATH:.. THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 jobman cmdline DARPAscript_simplified.NLPSDAE DARPA-bestrectifier_simplified.conf
PYTHONPATH=$PYTHONPATH:.. THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 jobman cmdline DARPAscript_simplified.NLPSDAE DARPA-bestrectifier_simplified.conf

