echo "-------------------"
echo "neural net, house, book, 16-8-2, .1"
bash nn_tester.sh house-votes-84 book '16,8,2' .1
echo "-------------------"
echo "neural net, house, glorot, 16-8-2, .1"
bash nn_tester.sh house-votes-84 glorot '16,8,2' .1

echo "-------------------"
echo "neural net, iris, book, 4-3, .1"
bash nn_tester.sh iris book '4,3' .1
echo "-------------------"
echo "neural net, iris, glorot, 4-3, .1"
bash nn_tester.sh iris glorot '4,3' .1

echo "-------------------"
echo "neural net, monks1, book, 6-2, .1"
bash nn_tester.sh monks1 book '6,2' .1
echo "-------------------"
echo "neural net, monks1, glorot, 6-2, .1"
bash nn_tester.sh monks1 glorot '6,2' .1

echo "-------------------this one won't work no matter what"
echo "neural net, monks2, book, 6-4-2, .1"
bash nn_tester.sh monks2 book '6,4,2' .1
echo "-------------------this one wont work no matter what"
echo "neural net, monks2, glorot, 6-4-2, .1"
bash nn_tester.sh monks2 glorot '6,4,2' .1

echo "-------------------"
echo "neural net, monks3, book, 6-2, .1"
bash nn_tester.sh monks3 book '6,2' .1
echo "-------------------"
echo "neural net, monks3, glorot, 6-2, .1"
bash nn_tester.sh monks3 glorot '6,2' .1
