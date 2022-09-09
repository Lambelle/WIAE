nO=1
nD=100
k_size=20
nI=50


lrD=0.0001
lrG=0.0001
batchsize=60
epochs=100

trainingC=10
gp_w=1.0
de_w=1.0
gp_w_ded=1.0

degree=4
block_size=100
strides=100

ts_perc=0.1667
data="MC.txt"
data_bad="MC_Anomaly_62.txt"

seed=140

python main.py -data $data -data_bad $data_bad -nO $nO -nD $nD -nI $nI -k_size $k_size -lrD $lrD -lrG $lrG -batchsize $batchsize -epochs $epochs -trainingC $trainingC -gp_W $gp_w -gp_W_ded $gp_w_ded -de_W $de_w -test_perc $ts_perc -degree $degree -block $block_size -stride $strides -seed $seed