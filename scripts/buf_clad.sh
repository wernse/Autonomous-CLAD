#!/usr/bin/bash

# nohup bash scripts/buf_clad.sh &
MY_PYTHON="python"
nb_seeds=2
seed=0
#source activate /data/wwon129
#cd /data/wwon129/HSIC

while [ $seed -le $nb_seeds ]
do


#  SGD
#  python main.py --gpu_id 2 --model sgd --teacher_backbone resnet18 --charlie 0 --alpha 0.1 --beta 10 --lr 0.03 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 10 --n_tasks 10 --buffer_size 1000 --name buf_tkd

#  JOINT
#  python main.py --gpu_id 1 --model joint --teacher_backbone resnet18 --charlie 0 --alpha 0.1 --beta 0.5 --lr 0.01 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 1 --n_tasks 10 --buffer_size 1000 --name buf_tkd

#  ER
#  python main.py --gpu_id 2 --model er_clad --lr 0.03 --buffer_size 200 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 10 --n_tasks 10 --name buf_tkd
#  python main.py --gpu_id 3 --model er_clad --lr 0.03 --buffer_size 500 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 10 --n_tasks 10 --name buf_tkd

#  ER + LSW
#  python main.py --gpu_id 2 --model er_clad_lsw --lr 0.03 --buffer_size 200 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 10 --n_tasks 10 --name buf_tkd
#  python main.py --gpu_id 3 --model er_clad_lsw --lr 0.03 --buffer_size 500 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 10 --n_tasks 10 --name buf_tkd

#  WSN
#  python main.py --gpu_id 2 --model wsn --teacher_backbone resnet18 --charlie 0 --alpha 0 --beta 0.5 --lr 0.03 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 10 --n_tasks 10 --buffer_size 200 --name buf_tkd
#  python main.py --gpu_id 2 --model wsn --teacher_backbone resnet18 --charlie 0 --alpha 0 --beta 0.5 --lr 0.03 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 10 --n_tasks 10 --buffer_size 500 --name buf_tkd

#  LEAF
#  python main.py --gpu_id 3 --model wsn --teacher_backbone resnet18 --charlie 0 --alpha 1 --beta 0.5 --lr 0.03 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 10 --n_tasks 10 --buffer_size 200 --name buf_tkd
#  python main.py --gpu_id 3 --model wsn --teacher_backbone resnet18 --charlie 0 --alpha 1 --beta 0.5 --lr 0.03 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 10 --n_tasks 10 --buffer_size 500 --name buf_tkd

#  DER++
#  python main.py --gpu_id 0 --model derpp_ablation_clad --teacher_backbone resnet18 --charlie 0 --alpha 0.1 --beta 0.5 --lr 0.03 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 10 --n_tasks 10 --buffer_size 200 --name buf_tkd
#  python main.py --gpu_id 0 --model derpp_ablation_clad --teacher_backbone resnet18 --charlie 0 --alpha 0.1 --beta 0.5 --lr 0.03 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 10 --n_tasks 10 --buffer_size 500 --name buf_tkd

#  DER++ + SATCH
#  python main.py --gpu_id 2 --model derpp_ablation_clad --teacher_backbone resnet18 --charlie 0.1 --tkd 1 --agreement 0 --plasticity 1 --stability 1 --alpha 0.1 --beta 0.5 --lr 0.03 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 10 --n_tasks 10 --buffer_size 200 --name buf_tkd
#  python main.py --gpu_id 1 --model derpp_ablation_clad --teacher_backbone resnet18 --charlie 0.1 --tkd 1 --agreement 0 --plasticity 1 --stability 1 --alpha 0.1 --beta 0.5 --lr 0.03 --dataset seq-clad --seed $seed --backbone resnet18_lg --epochs 10 --n_tasks 10 --buffer_size 500 --name buf_tkd

	((seed++))
done
