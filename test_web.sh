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
            --test_model_name QAre_DATASET_WebNLG_LR_1e-05_BS_8_0516_epoch_43_f1_0.5560.pickle \
            --max_len 150 \
            --rel_num 171 \
            --period 50 \
            --seed 12 \
            --gpus 0,1,2,3 \
            > web_056_test_0516.log 2>&1 &