import math
import os
import chess
import csv
import logging
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.99)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    ckpt_path = "./checkpoints/checkpoint" + ".pt"
    num_workers = 0
    log_interval_percent = 0.2  # Evaluar métricas cada tanto porciento

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config, name):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.name = name
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def configure_optimizer(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = raw_model.configure_optimizers(self.config)

    def run_split(self, split, epoch):
        """Corre un epoch completo sobre el split"""
        is_train = (split == 'train')
        self.model.train(is_train)
        data = self.train_dataset if is_train else self.test_dataset
        print(f"Verify Set: {len(data)}")
        loader = DataLoader(data, shuffle=is_train, pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)

        losses = []
        correct = 0
        total = 0

        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
        for it, (states, actions, rewards, fens, timesteps, goals) in pbar:
            states = states.to(self.device)
            actions = actions.to(self.device)
            goals = goals.to(self.device)
            targets = actions.clone()

            with torch.set_grad_enabled(is_train):

                logits, loss = self.model(states, actions, targets=targets, goals=goals, timesteps=timesteps)
                loss = loss.mean()
                losses.append(loss.item())

            preds = logits.argmax(dim=-1) # (B,T)
            correct += (preds == targets.squeeze(-1)).sum().item()
            total += targets.numel()

            if is_train:
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()

                if self.config.lr_decay:
                    pass

                pbar.set_description(f"Epoch {epoch+1}: Train Loss {loss.item():.5f}")

        avg_loss = float(np.mean(losses))
        accuracy = correct / total
        precision = accuracy
        recall = accuracy
        f1 = 2 * (precision * recall) / (precision + recall) if (precision+recall)>0 else 0.0

        return avg_loss, accuracy, precision, recall, f1

    def run_split_partial(self, split, max_steps=None):
        """Evalua el split hasta max_steps"""
        is_train = (split == 'train')
        self.model.train(is_train)
        data = self.train_dataset if is_train else self.test_dataset
        print(f"Verify Set: {len(data)}")
        loader = DataLoader(data, shuffle=is_train, pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)

        losses = []
        correct = 0
        total = 0

        steps = 0
        for it, (states, actions, rewards, fens, timesteps, goals) in enumerate(loader):
            states = states.to(self.device)
            actions = actions.to(self.device)
            goals = goals.to(self.device)
            targets = actions.clone()

            with torch.no_grad():
                logits, loss = self.model(states, actions, targets=targets, goals=goals, timesteps=timesteps)
                loss = loss.mean()
                losses.append(loss.item())

            preds = logits.argmax(dim=-1)
            correct += (preds == targets.squeeze(-1)).sum().item()
            total += targets.numel()
            steps += 1

            if max_steps is not None and steps >= max_steps:
                break

        avg_loss = float(np.mean(losses))
        accuracy = correct / total
        precision = accuracy
        recall = accuracy
        f1 = 2 * (precision * recall) / (precision + recall) if (precision+recall)>0 else 0.0
        return avg_loss, accuracy, precision, recall, f1

    def train(self):
        self.configure_optimizer()
        config = self.config
        model = self.model

        # Métricas finales
        epochs_list = []
        split_list = []
        loss_list = []
        acc_list = []
        prec_list = []
        rec_list = []
        f1_list = []

        # Métricas parciales (incluyendo test)
        partial_metrics = {
            'Epoch': [],
            'Split': [],
            'Loss': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-score': []
        }

        best_loss = float('inf')

        for epoch in range(config.max_epochs):
            train_loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=True,
                                      batch_size=config.batch_size, num_workers=config.num_workers)
            total_steps = len(train_loader)
            log_steps = max(1, int(total_steps * config.log_interval_percent))

            model.train(True)
            losses = []
            correct = 0
            total = 0
            loader_iter = iter(train_loader)
            for it in tqdm(range(total_steps)):
                states, actions, rewards, fens, timesteps, goals = next(loader_iter)
                states = states.to(self.device)
                actions = actions.to(self.device)
                goals = goals.to(self.device)
                targets = actions.clone()

                logits, loss = model(states, actions, targets=targets, goals=goals, timesteps=timesteps)
                loss = loss.mean()
                losses.append(loss.item())

                preds = logits.argmax(dim=-1)
                correct += (preds == targets.squeeze(-1)).sum().item()
                total += targets.numel()

                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()

                # Métricas parciales de train
                if (it+1) % log_steps == 0 or (it+1) == total_steps:
                    avg_loss = float(np.mean(losses))
                    accuracy = correct / total
                    precision = accuracy
                    recall = accuracy
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision+recall)>0 else 0.0

                    step_fraction = (it+1)/total_steps
                    partial_metrics['Epoch'].append(epoch + step_fraction)
                    partial_metrics['Split'].append('Train')
                    partial_metrics['Loss'].append(avg_loss)
                    partial_metrics['Accuracy'].append(accuracy)
                    partial_metrics['Precision'].append(precision)
                    partial_metrics['Recall'].append(recall)
                    partial_metrics['F1-score'].append(f1)

                    # Ahora evaluamos el test en el mismo punto
                    if self.test_dataset is not None:
                        test_loss, test_acc, test_prec, test_rec, test_f1 = self.run_split_partial('test')
                        partial_metrics['Epoch'].append(epoch + step_fraction)
                        partial_metrics['Split'].append('Test')
                        partial_metrics['Loss'].append(test_loss)
                        partial_metrics['Accuracy'].append(test_acc)
                        partial_metrics['Precision'].append(test_prec)
                        partial_metrics['Recall'].append(test_rec)
                        partial_metrics['F1-score'].append(test_f1)

            # Al terminar la época (train completo)
            train_avg_loss = float(np.mean(losses))
            train_accuracy = correct / total
            train_precision = train_accuracy
            train_recall = train_accuracy
            train_f1 = 2 * (train_precision * train_recall) / (train_precision + train_recall) if (train_precision+train_recall)>0 else 0.0

            epochs_list.append(epoch+1)
            split_list.append('Train')
            loss_list.append(train_avg_loss)
            acc_list.append(train_accuracy)
            prec_list.append(train_precision)
            rec_list.append(train_recall)
            f1_list.append(train_f1)

            # Evaluar test al final de la época también
            if self.test_dataset is not None:
                test_loss, test_acc, test_prec, test_rec, test_f1 = self.run_split('test', epoch)
                epochs_list.append(epoch+1)
                split_list.append('Test')
                loss_list.append(test_loss)
                acc_list.append(test_acc)
                prec_list.append(test_prec)
                rec_list.append(test_rec)
                f1_list.append(test_f1)
                logger.info(f"Test loss: {test_loss}")

                if test_loss < best_loss:
                    best_loss = test_loss
                    raw_model = model.module if hasattr(model, "module") else model
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_val_loss': best_loss,
                        'config': config,
                    }
                    torch.save(checkpoint, config.ckpt_path)
                    with open('checkpoints/model.pkl','wb') as f:
                        pickle.dump(raw_model, f)

        df_epoch = pd.DataFrame({
            'Epoch': epochs_list,
            'Split': split_list,
            'Loss': loss_list,
            'Accuracy': acc_list,
            'Precision': prec_list,
            'Recall': rec_list,
            'F1-score': f1_list
        })

        df_partial = pd.DataFrame(partial_metrics)
        df_all = pd.concat([df_epoch, df_partial], ignore_index=True).sort_values(by='Epoch')

        # Guardar CSV final
        file_exists = os.path.isfile('resultados_dt.csv')
        with open('resultados_dt.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['run', 'dataset', 'lr', 'n_head', 'dropout', 'weight_decay', 'n_layer', 'dimension', 'context_len', 'train_loss', 'test_loss', 'train_acc', 'test_acc', 'train_recall', 'test_recall', 'train_precision', 'test_precision', 'train_f1', 'test_f1'])
            # Último epoch
            last_epoch = config.max_epochs
            final_train_loss = df_epoch[(df_epoch['Epoch']==last_epoch)&(df_epoch['Split']=='Train')]['Loss'].values[0]
            final_test_loss = df_epoch[(df_epoch['Epoch']==last_epoch)&(df_epoch['Split']=='Test')]['Loss'].values[0] if 'Test' in df_epoch['Split'].values else None
            final_train_acc = df_epoch[(df_epoch['Epoch']==last_epoch)&(df_epoch['Split']=='Train')]['Accuracy'].values[0]
            final_test_acc = df_epoch[(df_epoch['Epoch']==last_epoch)&(df_epoch['Split']=='Test')]['Accuracy'].values[0] if 'Test' in df_epoch['Split'].values else None
            final_train_rec = df_epoch[(df_epoch['Epoch']==last_epoch)&(df_epoch['Split']=='Train')]['Recall'].values[0]
            final_test_rec = df_epoch[(df_epoch['Epoch']==last_epoch)&(df_epoch['Split']=='Test')]['Recall'].values[0] if 'Test' in df_epoch['Split'].values else None
            final_train_prec = df_epoch[(df_epoch['Epoch']==last_epoch)&(df_epoch['Split']=='Train')]['Precision'].values[0]
            final_test_prec = df_epoch[(df_epoch['Epoch']==last_epoch)&(df_epoch['Split']=='Test')]['Precision'].values[0] if 'Test' in df_epoch['Split'].values else None
            final_train_f1 = df_epoch[(df_epoch['Epoch']==last_epoch)&(df_epoch['Split']=='Train')]['F1-score'].values[0]
            final_test_f1 = df_epoch[(df_epoch['Epoch']==last_epoch)&(df_epoch['Split']=='Test')]['F1-score'].values[0] if 'Test' in df_epoch['Split'].values else None

            writer.writerow([self.name, self.name.split('_')[0], config.learning_rate, config.n_head, config.dropout, config.weight_decay, self.name.split('_')[1], self.name.split('_')[2], self.name.split('_')[3], 
                             round(final_train_loss, 4), round(final_test_loss, 4) if final_test_loss is not None else None, 
                             round(final_train_acc, 4), round(final_test_acc, 4) if final_test_acc is not None else None, 
                             round(final_train_rec, 4), round(final_test_rec, 4) if final_test_rec is not None else None, 
                             round(final_train_prec, 4), round(final_test_prec, 4) if final_test_prec is not None else None, 
                             round(final_train_f1, 4), round(final_test_f1, 4) if final_test_f1 is not None else None])

        sns.set(style="whitegrid")

        # Graficar con puntos intermedios
        plt.figure(figsize=(8,6))
        sns.lineplot(data=df_all, x='Epoch', y='Loss', hue='Split', marker='o')
        plt.title('Train & Test Loss (con steps)')
        plt.savefig(f'images/{self.name}_loss_curve_steps.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8,6))
        sns.lineplot(data=df_all, x='Epoch', y='Accuracy', hue='Split', marker='o')
        plt.title('Train & Test Accuracy (con steps)')
        plt.savefig(f'images/{self.name}_accuracy_curve_steps.png', dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8,6))
        sns.lineplot(data=df_all, x='Epoch', y='F1-score', hue='Split', marker='o')
        plt.title('Train & Test F1-Score (con steps)')
        plt.savefig(f'images/{self.name}_f1_curve_steps.png', dpi=300, bbox_inches='tight')
        plt.close()
