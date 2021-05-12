nohup python -u -m train --bert_dir bert-base-cased \
            --lr 1e-5 \
            --multi_gpu False \
            --dataset NYT \
            --batch_size 8 \
            --max_epoch 50 \
            --test_epoch 1 \
            --train_prefix train_triples \
            --dev_prefix dev_triples \
            --test_prefix test_triples \
            --max_len 150 \
            --rel_num 24 \
            --period 50 \
            --seed 71 \
            --theta 0.5 \
            > train0512.log 2>&1 &