nohup python -u -m train --bert_dir bert-base-cased \
            --lr 2e-5 \
            --multi_gpu True \
            --dataset NYT \
            --batch_size 8 \
            --max_epoch 100 \
            --test_epoch 3 \
            --train_prefix train_triples \
            --dev_prefix dev_triples \
            --test_prefix test_triples \
            --max_len 150 \
            --rel_num 24 \
            --period 50 \
            --seed 11 \
            --theta 0.5 \
            --neg_samp 10 \
            --gpus 4,5,6,7 \
            > train0516.log 2>&1 &