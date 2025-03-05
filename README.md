Used 'CNN based model' for DACON icon recogniton competition. Includes single model training code and ensemble code

"Parse for running"

Single model training : python -u train.py --learning_rate 0.001 --weight_decay 1e-5 --batch_size 32 --epochs 5 --optimizer "Adam" --loss_function "CrossEntropyLoss" --device "cpu" --train_csv "./datafiles/train.csv" --test_csv "./datafiles/test.csv" --submission_csv "./datafiles/submission.csv"

Ensemble : python -u ensemble.py --learning_rate 0.001 --weight_decay 1e-5 --batch_size 32 --epochs 5 --num_models 5 --optimizer "Adam" --device "cpu" --train_csv "./datafiles/train.csv" --test_csv "./datafiles/test.csv" --submission_csv "./datafiles/submission.csv"