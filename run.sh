export LR=2e-3 && export EPOCH=200 && WD=0 && DIR=output1

mkdir -p ./checkpoint/${DIR}/con
mkdir -p ./checkpoint/${DIR}/uncon

python run.py dinov2 ${LR} ${EPOCH} ${WD} imagenet2012_train_random_200k.pkl ${DIR} #-logit -norm
