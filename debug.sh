nohup python -u -m train --bert_dir bert-base-cased \
            --lr 1e-5 \
            --multi_gpu False \
            --dataset NYT \
            --batch_size 8 \
            --max_epoch 50 \
            --test_epoch 4 \
            --train_prefix train_triples \
            --dev_prefix dev_triples \
            --test_prefix test_triples \
            --max_len 150 \
            --rel_num 24 \
            --period 20 \
            --seed 12 \
            --theta 0.3 \
            --debug True \
            > debug0511.log 2>&1 &