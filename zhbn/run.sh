#CUDA_VISIBLE_DEVICES=0

python3 -u baseline_train.py \
  --task_name mmc \
  --train_file ../data/mmc_bn/filtered_56/splits/train.json \
  --valid_file ../data/mmc_bn/filtered_56/splits/valid.json \
  --test_file ../data/mmc_bn/filtered_56/splits/test.json \
  --do_train \
  --do_lower_case \
  --bert_model monsoon-nlp/muril-adapted-local \
  --max_seq_length 224 \
  --train_batch_size 64 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --output_dir './output'

#  --bert_model joeddav/xlm-roberta-large-xnli \ # prob
#  --bert_model neuralspace-reverie/indic-transformers-bn-distilbert \
#  --bert_model sagorsarker/bangla-bert-base \
#  --bert_model monsoon-nlp/muril-adapted-local \
#  --bert_model bert-base-uncased \
#  --bert_model neuralspace-reverie/indic-transformers-bn-roberta \
#  --bert_model neuralspace-reverie/indic-transformers-bn-bert \ # prob

#python -u demo.py \
#  --premise_str 'fuck why my email not come yet' \
#  --hypo_list 'anger | this text expresses anger | the guy is very unhappy'
