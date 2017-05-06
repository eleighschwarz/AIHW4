#!/bin/bash
#
# a shell script to automate testing. Makes batch testing simpler.
#

igList=(True False)
datasetList=(iris)
# monks1 monks2 monks3 house-votes-84)
pruneList=(True False)

echo "Running..."


for dataset in "${datasetList[@]}"; do
    echo "Running dataset..."
    for ig in "${igList[@]}"; do
	echo "Running ig..."
	for prune in "${pruneList[@]}"; do
	    echo "Running pruning..."
            echo "${dataset} | Ratio: ${ig} | Pruning: ${prune}" >> decision_tree_data.txt
            python classify.py --mode train --algorithm decision_tree --model-file ${dataset}_ratio${ig}_prune${prune} --data data/${dataset}.train --ratio ${ig} --prune ${prune}
	    python classify.py --mode test --algorithm decision_tree --model-file ${dataset}_ratio${ig}_prune${prune} --data data/${dataset}.test --ratio ${ig} --prune ${prune} > /dev/null
	    cat "${dataset}_ratio${ig}_prune${prune}_decision_tree_stats.txt" >> decision_tree_data.txt
	    echo "" >> decision_tree_data.txt
	done
    done
done
