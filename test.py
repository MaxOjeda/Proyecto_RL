import torch
import math
from torch.utils.data import DataLoader
from mingpt.model_chess import GPT, GPTConfig
import numpy as np
import pickle
from run_dt_chess import ChessDataset

def evaluate_model(model, dataset, batch_size=64, device='cpu'):
    # DataLoader para el test set
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    losses = []
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for states, actions, rewards, fens in loader:
            states = states.to(device)
            actions = actions.to(device)
            # calcular rtgs (suponiendo misma lógica que en entrenamiento)
            rtgs = torch.cumsum(rewards, dim=1).flip(dims=[1]).unsqueeze(-1).to(device)

            # targets son las acciones correctas a predecir
            targets = actions.clone()

            logits, loss = model(states, actions, targets=targets, rtgs=rtgs)
            loss = loss.mean()
            losses.append(loss.item())

            # Calcular top-1 y top-5 accuracy
            # logits: (B,T,vocab_size), targets: (B,T)
            # Tomamos la última predicción del bloque (depende cómo esté configurado el modelo)
            # Si el modelo predice acción a acción, podemos iterar sobre las secuencias.
            # Suponiendo que logits corresponde uno a uno con targets:
            # aplanamos para evaluar top-k en todo el batch
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)

            # top-1 accuracy
            preds_top1 = torch.argmax(logits_flat, dim=-1)
            correct_top1 += (preds_top1 == targets_flat).sum().item()

            # top-5 accuracy
            # Obtenemos los top-5 más probables
            top5 = torch.topk(logits_flat, k=5, dim=-1).indices  # (B*T,5)
            # Chequear si el target está en top5
            in_top5 = (top5 == targets_flat.unsqueeze(-1)).any(dim=-1)
            correct_top5 += in_top5.sum().item()

            total += targets_flat.size(0)

    avg_loss = np.mean(losses)
    perplexity = math.exp(avg_loss)
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    return avg_loss, perplexity, top1_acc, top5_acc

# Ejemplo de uso:
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Cargar dataset de prueba
    # Supongamos que ya tienes 'test_dataset' cargado.

    model_files = [
        "agents/chess_agent_8000_6_128_100",
        "agents/chess_agent_8000_6_128_150",
        "agents/chess_agent_8000_6_128_200",
        "agents/chess_agent_8000_6_256_150",
        "agents/chess_agent_8000_6_256_200",
        "agents/chess_agent_8000_12_128_150"
    ]

    # Ajustar vocab_size, context_length, n_layer, n_head, n_embd según tu entrenamiento.
    # Asegúrate de usar la misma configuración que usaste al entrenar cada modelo.
    # Si difieren, necesitas saber qué config se utilizó para cada modelo.
    # Por ejemplo (ajusta según tu setup real):
    vocab_size = 4096
    context_length = 100 # o el que corresponda a cada modelo
    n_layer = 6
    n_head = 8
    n_embd = 128
    model_type = 'reward_conditioned'

    # Cargar el test_dataset
    with open('data/processed/test_dataset_8000.pkl', 'rb') as f:
        test_data = pickle.load(f)
    test_dataset_100 = ChessDataset(data=test_data, block_size=100, model_type=model_type)
    test_dataset_150 = ChessDataset(data=test_data, block_size=150, model_type=model_type)
    test_dataset_200 = ChessDataset(data=test_data, block_size=200, model_type=model_type)

    results = []
    for model_file in model_files:
        with open(f'agents/configs/{model_file.split('/')[1]}.pkl','rb') as f:
            model_args = pickle.load(f)
        # Determinar la config a partir del nombre del modelo (si lo codificaste así)
        # Por ejemplo si el nombre del archivo tiene n_layer, n_embd en el nombre, parsea esos.
        # O simplemente define manualmente por ahora.

        # Crear el modelo
        mconf = GPTConfig(vocab_size=vocab_size, block_size=model_args.context_length*3,
                          n_layer=model_args.n_layers, n_head=model_args.n_head, n_embd=model_args.n_embd, model_type=model_type)
        model = GPT(mconf)
        state_dict = torch.load(model_file + ".pt", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        con_len = model_args.context_length
        if con_len == 100:
            test_dataset = test_dataset_100
        elif con_len == 150:
            test_dataset = test_dataset_150
        else:
            test_dataset = test_dataset_200

        loss, ppl, top1_acc, top5_acc = evaluate_model(model, test_dataset, batch_size=32, device=device)
        results.append((model_file, loss, ppl, top1_acc, top5_acc))
        print(f"Model: {model_file}\n  Test Loss: {loss}\n  Perplexity: {ppl}\n  Top-1 Accuracy: {top1_acc}\n  Top-5 Accuracy: {top5_acc}\n")

    # Finalmente puedes graficar loss, perplexity, accuracy con matplotlib
    import matplotlib.pyplot as plt
    # Ejemplo: graficar la pérdida de todos los modelos
    model_names = [r[0] for r in results]
    losses = [r[1] for r in results]
    plt.figure(figsize=(10,5))
    plt.bar(model_names, losses)
    plt.xticks(rotation=90)
    plt.ylabel("Test Loss")
    plt.title("Comparación de Pérdida en Test entre Agentes")
    plt.tight_layout()
    plt.savefig("test_loss_comparison.png")

    # Haz lo mismo con perplexity o accuracy si lo deseas.
