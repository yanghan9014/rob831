# Section 1
## Part 2
```
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --video_log_freq -1 –-ep_len 1000 --eval_batch_size 5000
```
## Part 3
```
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --video_log_freq -1 –-ep_len 1000 --eval_batch_size 5000
```
## Part 4
Repeat the experiment with varying `expert_data_usage`

```
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
    --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --video_log_freq -1 –-ep_len 1000 --eval_batch_size 5000 --expert_data_usage 0.9
```
Figure:
```
python rob831/scripts/plot_figure_ablation.py
```

# Section 2
## Part 2
```
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
    --video_log_freq -1 –-ep_len 1000 --eval_batch_size 5000
```
```
python rob831/scripts/plot_figures_dagger.py
```