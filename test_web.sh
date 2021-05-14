nohup python -u -m test --bert_dir bert-base-cased \
            --lr 1e-5 \
            --multi_gpu True \
            --dataset WebNLG \
            --batch_size 8 \
            --max_epoch 100 \
            --test_epoch 1 \
            --train_prefix train_triples \
            --dev_prefix dev_triples \
            --test_prefix test_triples \
            --test_model_name QAre_DATASET_WebNLG_LR_1e-05_BS_8_0512.pickle \
            --max_len 150 \
            --rel_num 171 \
            --period 50 \
            --seed 12 \
            --gpus 0,1,2,3 \
            > web_062_test_0513.log 2>&1 &