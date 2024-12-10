import torch
import os
import chess
import chess.pgn
import numpy as np
from mingpt.model_chess import GPT, GPTConfig
from create_dataset_chess import fen_to_tensor, move_to_index
from torch.nn import functional as F

def save_board_svg(board, move_number, directory="moves_svg"):
    os.makedirs(directory, exist_ok=True)
    svg_code = chess.svg.board(board=board, size=400)
    filename = os.path.join(directory, f"move_{move_number:03d}.svg")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(svg_code)
    print(f"Tablero guardado: {filename}")

# Función inversa a move_to_index para mapear indices a movimientos legales
def index_to_move(index):
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)

def load_model(model_path, vocab_size=4096, context_length=100, n_layer=6, n_head=8, n_embd=128):
    # Configuración del modelo
    mconf = GPTConfig(vocab_size=vocab_size, block_size=context_length*3,
                      n_layer=n_layer, n_head=n_head, n_embd=n_embd, model_type='reward_conditioned', max_timestep=220, dropout=0.1)
    model = GPT(mconf)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def choose_agent_move(model, board, context_length=100):
    # Convertir el estado actual a tensor
    fen = board.fen()
    state_tensor = fen_to_tensor(fen) # (8,8,12)
    state_tensor = torch.tensor(state_tensor, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    # state_tensor tiene dimensión (1,1,8,8,12): batch=1, block_size=1
    rtg = torch.zeros((1,1,1)) # rtg temp
    actions = None
    T = 1
    timesteps = torch.zeros((1,T), dtype=torch.int64)   # todo ceros
    with torch.no_grad():
        logits, _ = model(state_tensor, actions=actions, targets=None, rtgs=rtg, timesteps=timesteps)
        logits = logits[:,-1,:] # ultima posición
        probs = F.softmax(logits, dim=-1)

    legal_moves = list(board.legal_moves)
    # Mapear movimientos legales a indices
    legal_indices = [move_to_index(m) for m in legal_moves]
    # Filtrar probabilidades solo para índices legales
    legal_probs = probs[0, legal_indices]
    best_move_idx = legal_indices[torch.argmax(legal_probs).item()]

    best_move = index_to_move(best_move_idx)
    return best_move

if __name__ == "__main__":
    model = load_model("agents/chess_agent_8000_6_128_100_4_0.7_0.1.pt")

    # Iniciar un tablero
    board = chess.Board()

    print("Comencemos la partida. Eres las blancas (mueves primero).")
    print(board)
    move_number = 0
    while not board.is_game_over():
        # Movimiento humano
        human_move = input("Tu movimiento (ej: e2e4): ")
        try:
            move = chess.Move.from_uci(human_move)
            if move in board.legal_moves:
                board.push(move)
                print(board)
            else:
                print("Movimiento ilegal, intenta de nuevo.")
                continue
        except:
            print("Entrada inválida, intenta de nuevo.")
            continue

        if board.is_game_over():
            break

        # Movimiento del agente
        agent_move = choose_agent_move(model, board)
        print(f"El agente mueve: {agent_move}")
        board.push(agent_move)
        print(board)

        move_number += 1
        save_board_svg(board, move_number)

    print("Partida terminada. Resultado:", board.result())
