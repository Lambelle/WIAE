nO=1
nD=100
k_size=20
nI=50

lrD=0.001
lrG=0.001
batchsize=60
epochs=100

trainingC=10
gp_w=3
de_w=2
gp_w_ded=3

degree=4
block_size=200
strides=100

ts_perc=0.2
data="MC.txt"
data_bad="MC_Anomaly.txt"


python main.py -data $data -data_bad $data_bad -nO $nO -nD $nD -nI $nI -k_size $k_size -lrD $lrD -lrG $lrG -batchsize $batchsize -epochs $epochs -trainingC $trainingC -gp_W $gp_w -gp_W_ded $gp_w_ded -de_W $de_w -test_perc $ts_perc -degree $degree -block $block_size -stride $strides