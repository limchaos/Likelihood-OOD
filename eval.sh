export DIR=output1
mkdir -p ./eval/${DIR}
python eval.py dinov2 ${DIR} #-logit -norm
