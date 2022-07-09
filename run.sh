nO=1
nD=100
k_size=20
nI=50

lrD=0.0024
lrG=0.0024
batchsize=100
epochs=1

trainingC=10
gp_w=1
de_w=2
gp_w_ded=1

ts_perc=0.2
data="dataset/BESSZ.txt"
data_bad="dataset/BESSZ_bad.txt"
inn="result/BESSZ_inn.txt"
inn_test="result/BESSZ_inn_test.txt"
inn_bad="result/BESSZ_inn_bad.txt"

python main.py -data $data -data_bad $data_bad -inn $inn -inn_test $inn_test -inn_bad $inn_bad -nO $nO -nD $nD -nI $nI -k_size $k_size -lrD $lrD -lrG $lrG -batchsize $batchsize -epochs $epochs -trainingC $trainingC -gp_W $gp_w -gp_W_ded $gp_w_ded -de_W $de_w -test_perc $ts_perc