# Decision Transformer (DT)
for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 50 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
done

# Behavior Cloning (BC)
for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Breakout' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Qbert' --batch_size 128
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 50 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Pong' --batch_size 512
done

for seed in 123 231 312
do
    python run_dt_atari.py --seed $seed --context_length 30 --epochs 5 --model_type 'naive' --num_steps 500000 --num_buffers 50 --game 'Seaquest' --batch_size 128
done

python run_dt_chess.py --seed 123 --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 500000 --num_buffers 50 --batch_size 128

python run_dt_chess.py --context_length 100 --epochs 15 --batch_size 32 --n_embd 128
python run_dt_chess.py --context_length 100 --epochs 5 --batch_size 32 --n_embd 128 --n_layers 6 --n_head 8 --dataset 8000 --dropout 0.1 --weight_decay 0.1