import logging
import pickle
import torch
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset
from mingpt.model_chess import GPT, GPTConfig
from mingpt.trainer_chess import Trainer, TrainerConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

class ChessDataset(Dataset):
    def __init__(self, data, block_size, model_type='reward_conditioned'):
        self.data = data
        self.block_size = block_size
        self.model_type = model_type

        self.states = []
        self.actions = []
        self.rewards = []
        self.fens = []
        self.timesteps = []
        self.goals = []  # Nueva lista para goals

        for game in tqdm(data):
            states_game = game['states']
            actions_game = game['actions']
            rewards_game = game['rewards']
            fens_game = game['fens']
            timesteps_game = game['timesteps']
            goals_game = game['goals']  # extraemos goals

            if len(states_game) >= self.block_size:
                for i in range(len(states_game)-self.block_size):
                    self.states.append(np.array(states_game[i:i+self.block_size]))
                    self.actions.append(np.array(actions_game[i:i+self.block_size]).reshape(-1,1))
                    self.rewards.append(np.array(rewards_game[i:i+self.block_size]))
                    self.fens.append(fens_game[i:i+self.block_size])
                    self.timesteps.append(np.array(timesteps_game[i:i+self.block_size]))
                    self.goals.append(np.array(goals_game[i:i+self.block_size]).reshape(-1,1))  # guardar goals

        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.actions = torch.tensor(self.actions, dtype=torch.long)
        self.rewards = torch.tensor(self.rewards, dtype=torch.float32)
        self.timesteps = torch.tensor(self.timesteps, dtype=torch.int64)
        self.goals = torch.tensor(self.goals, dtype=torch.float32)  # convertir a tensor

        self.vocab_size = 4096

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (self.states[idx], 
                self.actions[idx], 
                self.rewards[idx], 
                self.fens[idx], 
                self.timesteps[idx], 
                self.goals[idx])  # retornar goals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--context_length', type=int, help="Context Length", default=50)
    parser.add_argument('--n_layers', type=int, help='Nombre del modelo', default=6)
    parser.add_argument('--n_head', type=int, help='Dataset Name', default=8)
    parser.add_argument('--n_embd', type=int, help='Dimensiones', default=128)
    parser.add_argument('--epochs', type=int, help='Number of epochs', default=10)
    parser.add_argument('--batch_size', type=int, help='Batch size', default=64)
    parser.add_argument('--lr', type=float, help='Learning rate', default=6e-4)
    parser.add_argument('--dataset', type=int, help='Learning rate', default=1000)
    parser.add_argument('--dropout', type=float, help='Drop out', default=0.1)
    parser.add_argument('--decay', type=bool, help='Decay', default=False)
    parser.add_argument('--weight_decay', type=float, help='Weight Decay', default=0.1)
    
    args = parser.parse_args()
    # Cargar datasets
    dataset_num = args.dataset
    print("Leer datos\n")
    with open(f'data/processed/train_dataset_{dataset_num}.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open(f'data/processed/test_dataset_{dataset_num}.pkl', 'rb') as f:
        test_data = pickle.load(f)

    # Parametros
    context_length = args.context_length
    n_layer = args.n_layers
    n_head = args.n_head
    n_embd = args.n_embd
    max_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    dropout = args.dropout
    weight_decay = args.weight_decay
    lr_decay = False

    print(f"Dataset Lenght: {dataset_num}")
    print(f"Epochs: {max_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Context Lenght: {context_length}")
    print(f"Emb Dimension: {n_embd}")
    print(f"Num Layers: {n_layer}")
    print(f"N Heads: {n_head}")
    print(f"Dropout: {dropout}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")
    vocab_size = 4096

    print("Preparar train dataset")
    train_exists = os.path.isfile(f'data/datasets/train_dataset_{dataset_num}_{context_length}.pkl')
    if not train_exists:
        train_dataset = ChessDataset(data=train_data, block_size=context_length, model_type='reward_conditioned')
        with open(f'data/datasets/train_dataset_{dataset_num}_{context_length}.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)
    else:
        print("Chess Train dataset exist! Loading...")
        with open(f'data/datasets/train_dataset_{dataset_num}_{context_length}.pkl', 'rb') as f:
            train_dataset = pickle.load(f)

    print("Preparar test dataset")
    test_exists = os.path.isfile(f'data/datasets/test_dataset_{dataset_num}_{context_length}.pkl')
    if not test_exists:
        test_dataset = ChessDataset(data=test_data, block_size=context_length, model_type='reward_conditioned')
        with open(f'data/datasets/test_dataset_{dataset_num}_{context_length}.pkl', 'wb') as f:
            pickle.dump(test_dataset, f)
    else:
        print("Chess Test dataset exist! Loading...")
        with open(f'data/datasets/test_dataset_{dataset_num}_{context_length}.pkl', 'rb') as f:
            test_dataset = pickle.load(f)

    #print(train_dataset.__getitem__(0))
    # print(f"state: {train_dataset.__getitem__(0)[0]}")
    # print(f"actions: {train_dataset.__getitem__(0)[1]}")
    # print(f"rewards: {train_dataset.__getitem__(0)[2]}")
    # print(f"timesteps: {train_dataset.__getitem__(0)[4]}")
    # print(f"goals: {train_dataset.__getitem__(0)[5]}")
    # input()
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    # Calcular max_timestep
    max_timestep = train_dataset.timesteps.max().item()
    max_timestep_test = test_dataset.timesteps.max().item()
    max_timestep = max(max_timestep, max_timestep_test)
    print("Max timestep:", max_timestep)

    print("Configuración GPT")
    mconf = GPTConfig(vocab_size=vocab_size, block_size=context_length*3,
                      n_layer=n_layer, n_head=n_head, dropout=dropout, n_embd=n_embd, model_type='reward_conditioned', max_timestep=max_timestep) # agregar timesteps
    
    print("Cargar modelo minGPT")
    model = GPT(mconf)
    #print(model)

    print("Configuración Trainer")
    tconf = TrainerConfig(max_epochs=max_epochs, batch_size=batch_size, learning_rate=learning_rate,
                          lr_decay=lr_decay, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*context_length*3,
                          num_workers=4, model_type='reward_conditioned', n_head=n_head, dropout=dropout, weight_decay=weight_decay)
    
    print("Trainer")
    name = f"{dataset_num}_{n_layer}_{n_embd}_{context_length}_{n_head}_{dropout}_{weight_decay}"
    trainer = Trainer(model, train_dataset, test_dataset, tconf, name)
    print("Entrenando...")
    trainer.train()

    print("Entrenamiento finalizado. Guardando el modelo...")
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(raw_model.state_dict(), f"agents/chess_agent_{name}.pt")
    with open(f'agents/configs/chess_agent_{name}.pkl','wb') as f:
                            pickle.dump(args, f)
