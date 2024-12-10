# data_preparation.py

import os
import chess
import chess.pgn
import numpy as np
import random
import pickle
from tqdm import tqdm

def fen_to_tensor(fen):
    """
    Convierte una cadena FEN en un tensor 8x8x12.
    Cada plano representa una de las 12 piezas posibles (6 tipos por 2 colores).
    """
    piece_map = {
        'P': 0,  'N': 1,  'B': 2,  'R': 3,  'Q': 4,  'K': 5,
        'p': 6,  'n': 7,  'b': 8,  'r': 9,  'q': 10, 'k': 11
    }
    board_tensor = np.zeros((8, 8, 12), dtype=np.float32)
    board = chess.Board(fen)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.symbol()
            plane = piece_map[piece_type]
            x = square % 8
            y = square // 8
            board_tensor[y, x, plane] = 1
    return board_tensor

def move_to_index(move):
    """
    Convierte un movimiento en formato UCI a un índice único.
    Hay 64x64 posibles movimientos (desde y hacia cada casilla).
    """
    from_square = move.from_square
    to_square = move.to_square
    index = from_square * 64 + to_square
    return index

def calculate_reward(result, turn):
    if result == "1-0":
        return 1 if turn == chess.WHITE else -1
    elif result == "0-1":
        return -1 if turn == chess.WHITE else 1
    else:
        return 0

def process_game(game):
    board = game.board()
    states = []
    actions = []
    rewards = []
    fens = []
    result = game.headers['Result']
    timesteps = []
    moves = list(game.mainline_moves())
    total_moves = len(moves)

    # Procesar movimientos
    timestep = 0
    for i, move in enumerate(moves):
        fen = board.fen()
        state_tensor = fen_to_tensor(fen)
        states.append(state_tensor)
        action_index = move_to_index(move)
        actions.append(action_index)
        board.push(move)
        # Recompensa basada en resultado
        reward = calculate_reward(result, board.turn)
        rewards.append(reward)
        timesteps.append(timestep)
        timestep += 1

    # goals = pasos restantes hasta el final
    goals = [total_moves - i for i in range(total_moves)]

    return {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'fens': fens,
        'timesteps': timesteps,
        'goals': goals
    }

def parse_pgn(pgn_file_path, max_games=None):
    # Procesar partidas
    games_data = []
    with open(pgn_file_path, 'r', encoding='utf-8') as pgn_file:
        game_count = 0
        with tqdm(total=max_games, desc="Procesando partidas") as pbar:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None or (max_games and game_count >= max_games):
                    break
                game_data = process_game(game)
                if len(game_data['states']) > 0: # Por si acaso
                    games_data.append(game_data)
                game_count += 1
                pbar.update(1)
    return games_data

def save_dataset(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Conjunto de datos guardado en {file_path}")

def main():
    max_games = 10000 
    pgn_file_path = 'data/raw/lichess_db_standard_rated_2014-06.pgn'
    train_dataset_path = f'data/processed/train_dataset_{max_games}.pkl'
    test_dataset_path = f'data/processed/test_dataset_{max_games}.pkl'

    # Procesar las partidas
    print("Leyendo y procesando partidas...")

    # Shuffle
    games_data = parse_pgn(pgn_file_path, max_games=max_games)
    random.shuffle(games_data)

    # Dividir entrenamiento y test
    split_ratio = 0.8
    split_idx = int(len(games_data) * split_ratio)
    train_data = games_data[:split_idx]
    test_data = games_data[split_idx:]

    # Guardar los datasets
    print("Guardando dataset..")
    save_dataset(train_data, train_dataset_path)
    save_dataset(test_data, test_dataset_path)
    print("¡Proceso completado!")

if __name__ == "__main__":
    main()
