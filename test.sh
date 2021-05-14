nohup python -u -m test --bert_dir bert-base-cased \
            --lr 1e-5 \
            --multi_gpu True \
            --dataset NYT \
            --batch_size 8 \
            --max_epoch 100 \
            --test_epoch 1 \
            --train_prefix train_triples \
            --dev_prefix dev_triples \
            --test_prefix test_triples \
            --test_model_name QAre_DATASET_NYT_LR_1e-05_BS_8_0512.pickle \
            --max_len 150 \
            --rel_num 24 \
            --period 50 \
            --seed 12 \
            --gpus 0,1,2,3 \
            > nyt_088_test_0512.log 2>&1 &