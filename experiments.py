import subprocess
import itertools

# Define los valores para los parámetros

context_lengths = ["100"]
dimensions = ["128"]
batch_sizes = ["32"]

# Genera todas las combinaciones posibles de los hiperparámetros
n_layers = ["6"]
n_heads = ["4", "8"]
dropouts = ["0.7"]
lrs = ["3e-4"]
weight_decays = ["0.1", "0.2"]
combinations = list(itertools.product(context_lengths, dimensions, n_layers, dropouts, lrs, n_heads, weight_decays))

# Itera sobre cada combinación y ejecuta el script con subprocess
for context_length, dim, n_layer, dropout, lr, head, w_decay in combinations:
    command = [
        "python", "run_dt_chess.py",
        "--context_length", str(context_length),
        "--epochs", "6",
        "--batch_size", "32",
        "--n_embd", str(dim),
        "--n_layers", str(n_layer),
        "--n_head", head,
        "--dataset", "8000",
        "--dropout", dropout,
        "--lr", lr,
        "--weight_decay", w_decay,
    ]
    
    print(f"Ejecutando: {command}")
    
    # Ejecuta el script
    subprocess.run(command)