#This script is for testing AI HW4 @JHU and generating statistics
#Created by Andrew Dykman
#Arguments can be edited to fit your customizations

DATA=$1
INIT=$2
SHAPE=$3
ALPHA=$4
MODEL="models/nn_${DATA}_alpha_1_epochs_100.model"
RESULT="results/raw/nn_${DATA}_test.results"

rm $MODEL
rm $RESULT

python classify.py --mode train --algorithm decision_tree --ig $IG --data data/$DATA.train --model-file $MODEL --nnshape ${SHAPE} --nnalpha $ALPHA --epochs 100 --nninitialization $INIT

python classify.py --mode test --algorithm decision_tree --ig $IG --data data/$DATA.test --model-file $MODEL >> $RESULT

python computeAccuracy.py --predicted $RESULT --actual data/$DATA.test
