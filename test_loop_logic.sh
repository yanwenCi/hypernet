#!/bin/bash
for ((i=0;i<=10; i++))
do
    num1=10-i
    for ((j=0;j<=num1;j++))
    do
        ((z=10-i-j))
        a=$(printf "%.2f" `echo "scale=3;$i/10"|bc`)
        b=$(printf "%.2f" `echo "scale=3;$j/10"|bc`)
        c=$(printf "%.2f" `echo "scale=3;$z/10"|bc`)
        echo "testing sequence $a, $b, $c"
        python test_hyper.py --img-list data/data_lesion/ --model-dir checkpoints/logistic_sum_soft  --mod 1 --load-weights 51 --gpu 1 --hyper_val $a,$b,$c   --pred-dir Pred_dir/results_logistic3
    done
done

