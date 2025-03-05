Used 'CNN based model' for DACON icon recogniton competition. Includes single model training code and ensemble code

"Parse for running"

Single model training : python -u train.py --learning_rate 0.001 --weight_decay 1e-5 --batch_size 32 --epochs 5 --optimizer "Adam" --loss_function "CrossEntropyLoss" --device "cpu"

Ensemble : python -u train.py --learning_rate 0.001 --weight_decay 1e-5 --batch_size 32 --epochs 5 --optimizer "Adam" --loss_function "CrossEntropyLoss" --device "cpu"