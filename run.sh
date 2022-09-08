nO=1
nD=100
k_size=20
nI=50


#lrD=0.0004
#lrG=0.0004
batchsize=60
epochs=100

trainingC=10
#gp_w=5
#de_w=0.6
#gp_w_ded=5

degree=4
block_size=100
strides=100

ts_perc=0.1667
data="MC.txt"
data_bad="MC_Anomaly_62.txt"

for lrD in $(seq 0.0001 0.0001 1)
do
  for lrG in $(seq 0.0001 0.0001 1)
  do
    for gp_w in $(seq 1 0.01 10)
    do
      for de_w in $(seq 1 0.01 10)
      do
        for gp_w_ded in $(seq 1 0.01 10)
        do
          for seed in $(seq 1 1 1000)
          do
            python main.py -data $data -data_bad $data_bad -nO $nO -nD $nD -nI $nI -k_size $k_size -lrD $lrD -lrG $lrG -batchsize $batchsize -epochs $epochs -trainingC $trainingC -gp_W $gp_w -gp_W_ded $gp_w_ded -de_W $de_w -test_perc $ts_perc -degree $degree -block $block_size -stride $strides -seed $seed
          done
        done
      done
    done
  done
done

echo "Training Finished!"