
# Official Implementation of Applying Continual Learning to Autononmous Driving Object Classification 

This codebase is based on [Mammoth](https://github.com/aimagelab/mammoth). The dataset is from [CLAD-D dataset](https://github.com/VerwimpEli/CLAD).

## 1. Experimental Results

### Class Incremental Learning

To reproduce the results reported for domain-incremental learning on CLAD-D, use the following commands:

```bash
bash scripts/buf_clad.sh
```

All hyperparameters can be seen in the bash files for the following methods:
- ER
- ER + LSW
- WSN
- LEAF
- DER++
- DER++ + SATCH

## 2. Example Command

Below is an example of how to run **DER++ with SATCH** on CLAD-D using a buffer size of 500:

```bash
python main.py --gpu_id 1 --model derpp_ablation_clad --teacher_backbone resnet18 --charlie 0.1 --tkd 1 --plasticity 1 --stability 1 --alpha 0.1 --beta 0.5 --lr 0.03 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 10 --n_tasks 10 --buffer_size 500 --name buf_tkd
```


