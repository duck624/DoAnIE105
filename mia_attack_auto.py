import os
import numpy as np
import torch
from collections import ChainMap
from sklearn import metrics
import matplotlib.pyplot as plt
import copy
import scipy
import time
import json
import math
import random
import sys

import warnings
warnings.filterwarnings('ignore')

print("Debug: Successfully imported all modules")

def liratio(mu_in, mu_out, var_in, var_out, new_samples):
    l_out = scipy.stats.norm.cdf(new_samples, mu_out, np.sqrt(var_out))
    return l_out

@torch.no_grad()
def hinge_loss_fn(x, y):
    x, y = copy.deepcopy(x).cuda(), copy.deepcopy(y).cuda()
    mask = torch.eye(x.shape[1], device="cuda")[y].bool()
    tmp1 = x[mask]
    x[mask] = -1e10
    tmp2 = torch.max(x, dim=1)[0]
    return (tmp1 - tmp2).cpu().numpy()

def ce_loss_fn(x, y):
    try:
        if x.dim() != 2 or y.dim() != 1:
            raise ValueError(f"Expected x to be 2D and y to be 1D, got x shape {x.shape} and y shape {y.shape}")
        
        min_samples = min(x.shape[0], y.shape[0])
        x = x[:min_samples]
        y = y[:min_samples]
        print(f"Debug: Adjusted shapes in ce_loss_fn - x: {x.shape}, y: {y.shape}")

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        return loss_fn(x, y)
    except Exception as e:
        print(f"Error in ce_loss_fn: {e}")
        return torch.tensor([])

def extract_hinge_loss(i):
    val_dict, train_dict, test_dict = {}, {}, {}
    try:
        val_index = i["val_index"]
        val_hinge_index = hinge_loss_fn(i["val_res"]["logit"], i["val_res"]["labels"])
        for j, k in zip(val_index, val_hinge_index):
            if j in val_dict:
                val_dict[j].append(k)
            else:
                val_dict[j] = [k]

        train_index = i["train_index"]
        train_hinge_index = hinge_loss_fn(i["train_res"]["logit"], i["train_res"]["labels"])
        for j, k in zip(train_index, train_hinge_index):
            if j in train_dict:
                train_dict[j].append(k)
            else:
                train_dict[j] = [k]

        test_index = i["test_index"]
        test_hinge_index = hinge_loss_fn(i["test_res"]["logit"], i["test_res"]["labels"])
        for j, k in zip(test_index, test_hinge_index):
            if j in test_dict:
                test_dict[j].append(k)
            else:
                test_dict[j] = [k]
    except Exception as e:
        print(f"Error in extract_hinge_loss: {e}")
    
    return val_dict, train_dict, test_dict

def plot_auc(name, target_val_score, target_train_score, epoch):
    if not torch.is_tensor(target_val_score) or not torch.is_tensor(target_train_score):
        print(f"Warning: Invalid input type in plot_auc for {name} at epoch {epoch}. Returning default values.")
        return 0, 0, {}
    
    if target_val_score.dim() == 0:
        target_val_score = target_val_score.unsqueeze(0)
    if target_train_score.dim() == 0:
        target_train_score = target_train_score.unsqueeze(0)

    target_val_score = target_val_score.cpu().numpy()
    target_train_score = target_train_score.cpu().numpy()

    if len(target_val_score) == 0 or len(target_train_score) == 0:
        print(f"Warning: Empty data in plot_auc for {name} at epoch {epoch}. Returning default values.")
        return 0, 0, {}

    labels = np.concatenate([np.zeros_like(target_val_score), np.ones_like(target_train_score)])
    scores = np.concatenate([target_val_score, target_train_score])

    if len(scores) < 2 or len(np.unique(labels)) < 2:
        print(f"Warning: Not enough data to compute ROC for {name} at epoch {epoch}. Returning default values.")
        return 0, 0, {}

    # Kiểm tra NaN trong scores
    if np.any(np.isnan(scores)):
        print(f"Error: Scores contain NaN in plot_auc for {name} at epoch {epoch}. Returning default values.")
        return 0, 0, {}

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)
    log_tpr, log_fpr = np.log10(tpr), np.log10(fpr)
    log_tpr[log_tpr < -5] = -5
    log_fpr[log_fpr < -5] = -5
    log_fpr = (log_fpr + 5) / 5.0
    log_tpr = (log_tpr + 5) / 5.0
    log_auc = metrics.auc(log_fpr, log_tpr)

    tprs = {}
    for fpr_thres in [10, 1, 0.1, 0.02, 0.01, 0.001, 0.0001]:
        tpr_index = np.sum(fpr < fpr_thres)
        tprs[str(fpr_thres)] = tpr[tpr_index - 1] if tpr_index > 0 else 0
    
    return auc, log_auc, tprs

def common_attack(f, K, epch, extract_fn=None):
    accs = []
    try:
        target_res = torch.load(f.format(0, epch))
        print(f"Debug: Successfully loaded target file {f.format(0, epch)} with keys {target_res.keys()}")
    except Exception as e:
        print(f"Error loading file {f.format(0, epch)}: {e}")
        return accs, {}, 0, 0, (np.array([]), np.array([]))

    if "train_res" not in target_res or "logit" not in target_res["train_res"] or "labels" not in target_res["train_res"]:
        print(f"Error: Missing train_res/logit/labels in target file for epoch {epch}")
        return accs, {}, 0, 0, (np.array([]), np.array([]))

    target_train_loss = -ce_loss_fn(target_res["train_res"]["logit"], target_res["train_res"]["labels"])
    if target_train_loss.numel() == 0:
        print(f"Error: Failed to compute target_train_loss for epoch {epch}")
        return accs, {}, 0, 0, (np.array([]), np.array([]))

    if MODE == "test":
        if "test_res" not in target_res or "logit" not in target_res["test_res"] or "labels" not in target_res["test_res"]:
            print(f"Error: Missing test_res/logit/labels in target file for epoch {epch}")
            return accs, {}, 0, 0, (np.array([]), np.array([]))
        target_test_loss = -ce_loss_fn(target_res["test_res"]["logit"], target_res["test_res"]["labels"])
    elif MODE == "val":
        if "val_res" not in target_res or "logit" not in target_res["val_res"] or "labels" not in target_res["val_res"]:
            print(f"Error: Missing val_res/logit/labels in target file for epoch {epch}")
            return accs, {}, 0, 0, (np.array([]), np.array([]))
        target_test_loss = -ce_loss_fn(target_res["val_res"]["logit"], target_res["val_res"]["labels"])
    else:
        print(f"Error: Invalid MODE {MODE} for epoch {epch}")
        return accs, {}, 0, 0, (np.array([]), np.array([]))

    if target_test_loss.numel() == 0:
        print(f"Error: Failed to compute target_test_loss for epoch {epch}")
        return accs, {}, 0, 0, (np.array([]), np.array([]))

    if target_train_loss.dim() == 0:
        target_train_loss = target_train_loss.unsqueeze(0)
    if target_test_loss.dim() == 0:
        target_test_loss = target_test_loss.unsqueeze(0)

    auc, log_auc, tprs = plot_auc("common", target_test_loss, target_train_loss, epch)
    print(f"__{'_' * 10} common")
    print(f"Debug: tprs = {tprs}, log_auc = {log_auc}")
    print(f"__{'_' * 10}")

    return accs, tprs, auc, log_auc, (target_test_loss.cpu().numpy(), target_train_loss.cpu().numpy())

def lira_attack_ldh_cosine(f, epch, K, save_dir, extract_fn=None, attack_mode="cos"):
    print('******************************************************')
    print('************', 'Epch:', epch, ' attack_mode:', attack_mode, '**************')
    print('******************************************************')
    save_log = save_dir + '/' + f'attack_sel{select_mode}_{select_method}_{attack_mode}.log'
    accs = []
    training_res = []

    if not isinstance(f, str):
        print(f"Error: f is not a string, got type {type(f)}. Expected a path template.")
        return accs, {}, 0, 0, (np.array([]), np.array([]))

    for i in range(K):
        file_path = f.format(i, epch)
        try:
            print(f"Debug: Attempting to load file {file_path}")
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} does not exist.")
                return accs, {}, 0, 0, (np.array([]), np.array([]))
            res = torch.load(file_path)
            print(f"Debug: Successfully loaded file {file_path} with keys {res.keys()}")
            training_res.append(res)
            accs.append(training_res[-1]["test_acc"])
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return accs, {}, 0, 0, (np.array([]), np.array([]))

    target_idx = 0
    val_idx = 1
    target_res = training_res[target_idx]
    shadow_res = training_res[val_idx:]

    if attack_mode == "cos":
        try:
            target_train_loss = torch.tensor(target_res.get("train_cos", target_res["train_res"].get("cos", None)))
            if target_train_loss is None or target_train_loss.numel() == 0:
                raise KeyError("train_cos is None or empty")
            print(f"Debug: target_train_loss shape (cos): {target_train_loss.shape}")
            
            if MODE == "test":
                target_test_loss = torch.tensor(target_res.get("test_cos", target_res["test_res"].get("cos", None)))
            elif MODE == "val":
                target_test_loss = torch.tensor(target_res.get("val_cos", target_res["val_res"].get("cos", None)))
            elif MODE == 'mix':
                if "mix_cos" not in target_res:
                    print("Warning: 'mix_cos' not found, falling back to MODE='val'.")
                    target_test_loss = torch.tensor(target_res.get("val_cos", target_res["val_res"].get("cos", None)))
                else:
                    random_indices = torch.randperm(target_res["test_cos"].shape[0])
                    target_test_loss = target_res["test_cos"][random_indices[:mix_length]]
                    target_test_loss = torch.tensor(target_test_loss)
                    mix_test_loss = torch.tensor(target_res["mix_cos"])
                    mix_test_loss = torch.cat([target_test_loss, mix_test_loss], dim=0)
                    target_test_loss = mix_test_loss
            if target_test_loss is None or target_test_loss.numel() == 0:
                raise KeyError("target_test_loss is None or empty")
            print(f"Debug: target_test_loss shape (cos): {target_test_loss.shape}")
        except KeyError as e:
            print(f"Warning: {e}, using loss from 'train_res' instead.")
            target_train_loss = -ce_loss_fn(target_res["train_res"]["logit"], target_res["train_res"]["labels"])
            if MODE == "test":
                target_test_loss = -ce_loss_fn(target_res["test_res"]["logit"], target_res["test_res"]["labels"])
            elif MODE == "val":
                target_test_loss = -ce_loss_fn(target_res["val_res"]["logit"], target_res["val_res"]["labels"])
            elif MODE == 'mix':
                if "mix_res" not in target_res:
                    print("Warning: 'mix_res' not found, falling back to MODE='val'.")
                    target_test_loss = -ce_loss_fn(target_res["val_res"]["logit"], target_res["val_res"]["labels"])
                else:
                    random_indices = torch.randperm(target_res["test_res"]["logit"].shape[0])
                    target_test_loss = -ce_loss_fn(target_res["test_res"]["logit"][random_indices[:mix_length]], target_res["test_res"]["labels"][random_indices[:mix_length]])
                    target_test_loss = torch.tensor(target_test_loss)
                    mix_test_loss = -ce_loss_fn(target_res["mix_res"]["logit"], target_res["mix_res"]["labels"])
                    mix_test_loss = torch.cat([target_test_loss, mix_test_loss], dim=0)
                    target_test_loss = mix_test_loss

    elif attack_mode == "diff":
        try:
            target_train_loss = torch.tensor(target_res.get("train_diffs", target_res["train_res"].get("diffs", None)))
            if target_train_loss is None or target_train_loss.numel() == 0:
                raise KeyError("train_diffs is None or empty")
            print(f"Debug: target_train_loss (before norm) min: {target_train_loss.min().item()}, max: {target_train_loss.max().item()}")
            target_train_loss = (target_train_loss - target_train_loss.mean()) / (target_train_loss.std() + 1e-8)
            print(f"Debug: target_train_loss shape (diff): {target_train_loss.shape}")
            print(f"Debug: target_train_loss mean (diff): {target_train_loss.mean().item()}, var: {target_train_loss.var().item()}")
            
            if MODE == "test":
                target_test_loss = torch.tensor(target_res.get("test_diffs", target_res["test_res"].get("diffs", None)))
            elif MODE == "val":
                target_test_loss = torch.tensor(target_res.get("val_diffs", target_res["val_res"].get("diffs", None)))
            if target_test_loss is None or target_test_loss.numel() == 0:
                raise KeyError("target_test_loss is None or empty")
            print(f"Debug: target_test_loss (before norm) min: {target_test_loss.min().item()}, max: {target_test_loss.max().item()}")
            target_test_loss = (target_test_loss - target_test_loss.mean()) / (target_test_loss.std() + 1e-8)
            print(f"Debug: target_test_loss shape (diff): {target_test_loss.shape}")
            print(f"Debug: target_test_loss mean (diff): {target_test_loss.mean().item()}, var: {target_test_loss.var().item()}")
        except KeyError as e:
            print(f"Warning: {e}, using loss from 'train_res' instead.")
            target_train_loss = -ce_loss_fn(target_res["train_res"]["logit"], target_res["train_res"]["labels"])
            if MODE == "test":
                target_test_loss = -ce_loss_fn(target_res["test_res"]["logit"], target_res["test_res"]["labels"])
            elif MODE == "val":
                target_test_loss = -ce_loss_fn(target_res["val_res"]["logit"], target_res["val_res"]["labels"])

    elif attack_mode == 'loss':
        try:
            target_train_loss = -ce_loss_fn(target_res["train_res"]["logit"], target_res["train_res"]["labels"])
            if target_train_loss.numel() == 0:
                raise ValueError("Failed to compute target_train_loss")
            print(f"Debug: target_train_loss (before log) min: {target_train_loss.min().item()}, max: {target_train_loss.max().item()}")
            target_train_loss = torch.clamp(target_train_loss, min=0)
            target_train_loss = torch.log(target_train_loss + 1e-8)
            print(f"Debug: target_train_loss shape (loss): {target_train_loss.shape}")
            print(f"Debug: target_train_loss mean (loss): {target_train_loss.mean().item()}, var: {target_train_loss.var().item()}")
            
            if MODE == "test":
                target_test_loss = -ce_loss_fn(target_res["test_res"]["logit"], target_res["test_res"]["labels"])
            elif MODE == "val":
                target_test_loss = -ce_loss_fn(target_res["val_res"]["logit"], target_res["val_res"]["labels"])
            elif MODE == 'mix':
                if "mix_res" not in target_res:
                    print("Warning: 'mix_res' not found, falling back to MODE='val'.")
                    target_test_loss = -ce_loss_fn(target_res["val_res"]["logit"], target_res["val_res"]["labels"])
                else:
                    random_indices = torch.randperm(target_res["test_res"]["logit"].shape[0])
                    target_test_loss = -ce_loss_fn(target_res["test_res"]["logit"][random_indices[:mix_length]], target_res["test_res"]["labels"][random_indices[:mix_length]])
                    target_test_loss = torch.tensor(target_test_loss)
                    mix_test_loss = -ce_loss_fn(target_res["mix_res"]["logit"], target_res["mix_res"]["labels"])
                    mix_test_loss = torch.cat([target_test_loss, mix_test_loss], dim=0)
                    target_test_loss = mix_test_loss
            if target_test_loss.numel() == 0:
                raise ValueError("Failed to compute target_test_loss")
            print(f"Debug: target_test_loss (before log) min: {target_test_loss.min().item()}, max: {target_test_loss.max().item()}")
            target_test_loss = torch.clamp(target_test_loss, min=0)
            target_test_loss = torch.log(target_test_loss + 1e-8)
            print(f"Debug: target_test_loss shape (loss): {target_test_loss.shape}")
            print(f"Debug: target_test_loss mean (loss): {target_test_loss.mean().item()}, var: {target_test_loss.var().item()}")
        except Exception as e:
            print(f"Error in loss mode: {e}")
            return accs, {}, 0, 0, (np.array([]), np.array([]))

    else:
        print(f"Error: Unsupported attack_mode {attack_mode}")
        return accs, {}, 0, 0, (np.array([]), np.array([]))

    if target_train_loss.dim() == 0:
        target_train_loss = target_train_loss.unsqueeze(0)
    if target_test_loss.dim() == 0:
        target_test_loss = target_test_loss.unsqueeze(0)

    shadow_train_losses = []
    shadow_test_losses = []
    if attack_mode == "cos":
        for i in shadow_res:
            try:
                shadow_train_loss = torch.tensor(i.get("train_cos", i["train_res"].get("cos", None)))
                if shadow_train_loss is None or shadow_train_loss.numel() == 0:
                    raise KeyError("shadow_train_loss is None or empty")
                
                if MODE == "val":
                    shadow_test_loss = torch.tensor(i.get("val_cos", i["val_res"].get("cos", None)))
                elif MODE == "test":
                    shadow_test_loss = torch.tensor(i.get("test_cos", i["test_res"].get("cos", None)))
                elif MODE == 'mix':
                    if "mix_cos" not in i:
                        print("Warning: 'mix_cos' not found in shadow_res, falling back to MODE='val'.")
                        shadow_test_loss = torch.tensor(i.get("val_cos", i["val_res"].get("cos", None)))
                    else:
                        random_indices = torch.randperm(i["test_cos"].shape[0])
                        shadow_test_loss = i["test_cos"][random_indices[:mix_length]]
                        shadow_test_loss = torch.tensor(shadow_test_loss)
                        mix_test_loss = torch.tensor(i["mix_cos"])
                        mix_test_loss = torch.cat([shadow_test_loss, mix_test_loss], dim=0)
                        shadow_test_loss = mix_test_loss
                if shadow_test_loss is None or shadow_test_loss.numel() == 0:
                    raise KeyError("shadow_test_loss is None or empty")
            except KeyError as e:
                print(f"Warning: {e} in shadow_res, using loss from 'train_res' instead.")
                shadow_train_loss = -ce_loss_fn(i["train_res"]["logit"], i["train_res"]["labels"])
                if MODE == "val":
                    shadow_test_loss = -ce_loss_fn(i["val_res"]["logit"], i["val_res"]["labels"])
                elif MODE == "test":
                    shadow_test_loss = -ce_loss_fn(i["test_res"]["logit"], i["test_res"]["labels"])
                elif MODE == 'mix':
                    if "mix_res" not in i:
                        print("Warning: 'mix_res' not found in shadow_res, falling back to MODE='val'.")
                        shadow_test_loss = -ce_loss_fn(i["val_res"]["logit"], i["val_res"]["labels"])
                    else:
                        random_indices = torch.randperm(i["test_res"]["logit"].shape[0])
                        shadow_test_loss = -ce_loss_fn(i["test_res"]["logit"][random_indices[:mix_length]], i["test_res"]["labels"][random_indices[:mix_length]])
                        shadow_test_loss = torch.tensor(shadow_test_loss)
                        mix_test_loss = -ce_loss_fn(i["mix_res"]["logit"], i["mix_res"]["labels"])
                        mix_test_loss = torch.cat([shadow_test_loss, mix_test_loss], dim=0)
                        shadow_test_loss = mix_test_loss

            if shadow_train_loss.dim() == 0:
                shadow_train_loss = shadow_train_loss.unsqueeze(0)
            if shadow_test_loss.dim() == 0:
                shadow_test_loss = shadow_test_loss.unsqueeze(0)

            shadow_train_losses.append(shadow_train_loss.cpu().numpy())
            shadow_test_losses.append(shadow_test_loss.cpu().numpy())
    elif attack_mode == "diff":
        for i in shadow_res:
            try:
                shadow_train_loss = torch.tensor(i.get("train_diffs", i["train_res"].get("diffs", None)))
                if shadow_train_loss is None or shadow_train_loss.numel() == 0:
                    print(f"Warning: Skipping client with empty train_diffs")
                    continue
                shadow_train_loss = (shadow_train_loss - shadow_train_loss.mean()) / (shadow_train_loss.std() + 1e-8)
                
                if MODE == "val":
                    shadow_test_loss = torch.tensor(i.get("val_diffs", i["val_res"].get("diffs", None)))
                elif MODE == "test":
                    shadow_test_loss = torch.tensor(i.get("test_diffs", i["test_res"].get("diffs", None)))
                if shadow_test_loss is None or shadow_test_loss.numel() == 0:
                    print(f"Warning: Skipping client with empty val_diffs/test_diffs")
                    continue
                shadow_test_loss = (shadow_test_loss - shadow_test_loss.mean()) / (shadow_test_loss.std() + 1e-8)

                if shadow_train_loss.dim() == 0:
                    shadow_train_loss = shadow_train_loss.unsqueeze(0)
                if shadow_test_loss.dim() == 0:
                    shadow_test_loss = shadow_test_loss.unsqueeze(0)

                shadow_train_losses.append(shadow_train_loss.cpu().numpy())
                shadow_test_losses.append(shadow_test_loss.cpu().numpy())
            except Exception as e:
                print(f"Warning: Error processing shadow_res client: {e}. Skipping this client.")
    elif attack_mode == "loss":
        for i in shadow_res:
            try:
                shadow_train_loss = -ce_loss_fn(i["train_res"]["logit"], i["train_res"]["labels"])
                if shadow_train_loss.numel() == 0:
                    print(f"Warning: Skipping client with empty train loss")
                    continue
                shadow_train_loss = torch.clamp(shadow_train_loss, min=0)
                shadow_train_loss = torch.log(shadow_train_loss + 1e-8)
                
                if MODE == "val":
                    shadow_test_loss = -ce_loss_fn(i["val_res"]["logit"], i["val_res"]["labels"])
                elif MODE == "test":
                    shadow_test_loss = -ce_loss_fn(i["test_res"]["logit"], i["test_res"]["labels"])
                if shadow_test_loss.numel() == 0:
                    print(f"Warning: Skipping client with empty val/test loss")
                    continue
                shadow_test_loss = torch.clamp(shadow_test_loss, min=0)
                shadow_test_loss = torch.log(shadow_test_loss + 1e-8)

                if shadow_train_loss.dim() == 0:
                    shadow_train_loss = shadow_train_loss.unsqueeze(0)
                if shadow_test_loss.dim() == 0:
                    shadow_test_loss = shadow_test_loss.unsqueeze(0)

                shadow_train_losses.append(shadow_train_loss.cpu().numpy())
                shadow_test_losses.append(shadow_test_loss.cpu().numpy())
            except Exception as e:
                print(f"Warning: Error processing shadow_res client: {e}. Skipping this client.")

    if shadow_train_losses:
        min_train_size = min(arr.shape[0] for arr in shadow_train_losses if arr.size > 0)
        shadow_train_losses = [arr[:min_train_size] for arr in shadow_train_losses if arr.size > 0]
        shadow_train_losses_stack = np.vstack(shadow_train_losses) if shadow_train_losses else np.array([])
        print(f"Debug: shadow_train_losses_stack shape: {shadow_train_losses_stack.shape}")
    else:
        shadow_train_losses_stack = np.array([])
        print("Debug: shadow_train_losses_stack is empty")

    if shadow_test_losses:
        min_test_size = min(arr.shape[0] for arr in shadow_test_losses if arr.size > 0)
        shadow_test_losses = [arr[:min_test_size] for arr in shadow_test_losses if arr.size > 0]
        shadow_test_losses_stack = np.vstack(shadow_test_losses) if shadow_test_losses else np.array([])
        print(f"Debug: shadow_test_losses_stack shape: {shadow_test_losses_stack.shape}")
    else:
        shadow_test_losses_stack = np.array([])
        print("Debug: shadow_test_losses_stack is empty")

    target_train_loss = target_train_loss.cpu().numpy()
    target_test_loss = target_test_loss.cpu().numpy()

    if shadow_train_losses_stack.size > 0 and target_train_loss.size > 0:
        min_train_size = min(target_train_loss.shape[0], min_train_size)
        target_train_loss = target_train_loss[:min_train_size]
    if shadow_test_losses_stack.size > 0 and target_test_loss.size > 0:
        min_test_size = min(target_test_loss.shape[0], min_test_size)
        target_test_loss = target_test_loss[:min_test_size]

    print('mean 0 \t train:', target_train_loss.mean() if target_train_loss.size > 0 else 'N/A', 
          '\tvar:', target_train_loss.var() if target_train_loss.size > 0 else 'N/A', 
          ' \t test:', target_test_loss.mean() if target_test_loss.size > 0 else 'N/A', 
          '\tvar:', target_test_loss.var() if target_test_loss.size > 0 else 'N/A')

    if attack_mode != 'cos' or select_mode == 0 or (select_method != 'mean_per' and select_method != 'outlier'):
        train_mu_out = shadow_train_losses_stack.mean(axis=0) if shadow_train_losses_stack.size > 0 else np.array([])
        train_var_out = (shadow_train_losses_stack.var(axis=0) + 1e-8) if shadow_train_losses_stack.size > 0 else np.array([])
        test_mu_out = shadow_test_losses_stack.mean(axis=0) if shadow_test_losses_stack.size > 0 else np.array([])
        test_var_out = (shadow_test_losses_stack.var(axis=0) + 1e-8) if shadow_test_losses_stack.size > 0 else np.array([])

    if train_mu_out.size == 0 or test_mu_out.size == 0:
        print("Warning: Shadow data is empty, using target data statistics as fallback.")
        train_mu_out = target_train_loss.mean() if target_train_loss.size > 0 else 0
        train_var_out = target_train_loss.var() + 1e-8 if target_train_loss.size > 0 else 1e-8
        test_mu_out = target_test_loss.mean() if target_test_loss.size > 0 else 0
        test_var_out = target_test_loss.var() + 1e-8 if target_test_loss.size > 0 else 1e-8

    train_l_out = scipy.stats.norm.cdf(target_train_loss, train_mu_out, np.sqrt(train_var_out)) if target_train_loss.size > 0 else np.array([])
    test_l_out = scipy.stats.norm.cdf(target_test_loss, test_mu_out, np.sqrt(test_var_out)) if target_test_loss.size > 0 else np.array([])
    print('var of train:', np.sqrt(train_var_out).mean() if np.isscalar(train_var_out) else np.sqrt(train_var_out).mean(), 
          'var of test:', np.sqrt(test_var_out).mean() if np.isscalar(test_var_out) else np.sqrt(test_var_out).mean())
    print("attack_mode:", attack_mode)
    print("mean of train_l_out:", train_l_out.mean() if train_l_out.size > 0 else 'N/A', 
          "var of train_l_out:", train_l_out.var() if train_l_out.size > 0 else 'N/A')
    print("mean of test_l_out:", test_l_out.mean() if test_l_out.size > 0 else 'N/A', 
          "var of test_l_out:", test_l_out.var() if test_l_out.size > 0 else 'N/A')
    print('test_l_out shape:', test_l_out.shape)

    if np.any(np.isnan(train_l_out)) or np.any(np.isnan(test_l_out)):
        print("Error: train_l_out or test_l_out contains NaN. Skipping AUC calculation.")
        return accs, {}, 0, 0, (train_l_out, test_l_out)

    auc, log_auc, tprs = plot_auc("lira", torch.tensor(test_l_out), torch.tensor(train_l_out), epch) if test_l_out.size > 0 and train_l_out.size > 0 else (0, 0, {})
    if epch % 10 == 0:
        print(f"__{'_' * 10} lira_attack")
        print(f"Debug: tprs = {tprs}, log_auc = {log_auc}")

    return accs, tprs, auc, log_auc, (train_l_out, test_l_out)

def cos_attack(f, K, epch, extract_fn=None):
    accs = []
    try:
        target_res = torch.load(f.format(0, epch))
        print(f"Debug: Successfully loaded target file {f.format(0, epch)} with keys {target_res.keys()}")
    except Exception as e:
        print(f"Error loading file {f.format(0, epch)}: {e}")
        return accs, {}, 0, 0, (np.array([]), np.array([]))

    target_train_loss = torch.tensor(target_res.get("train_cos", target_res["train_res"].get("cos", None)))
    if target_train_loss is None or target_train_loss.numel() == 0:
        print("Warning: 'train_cos' not found or empty, using loss from 'train_res' instead.")
        target_train_loss = -ce_loss_fn(target_res["train_res"]["logit"], target_res["train_res"]["labels"])
    
    if target_train_loss.numel() == 0:
        print(f"Error: Failed to compute target_train_loss for epoch {epch}")
        return accs, {}, 0, 0, (np.array([]), np.array([]))
    print(f"Debug: target_train_loss shape (cos_attack): {target_train_loss.shape}")

    if MODE == "test":
        target_test_loss = torch.tensor(target_res.get("test_cos", target_res["test_res"].get("cos", None)))
        if target_test_loss is None or target_test_loss.numel() == 0:
            print("Warning: 'test_cos' not found or empty, using loss from 'test_res' instead.")
            target_test_loss = -ce_loss_fn(target_res["test_res"]["logit"], target_res["test_res"]["labels"])
    elif MODE == "val":
        target_test_loss = torch.tensor(target_res.get("val_cos", target_res["val_res"].get("cos", None)))
        if target_test_loss is None or target_test_loss.numel() == 0:
            print("Warning: 'val_cos' not found or empty, using loss from 'val_res' instead.")
            target_test_loss = -ce_loss_fn(target_res["val_res"]["logit"], target_res["val_res"]["labels"])
    else:
        print(f"Error: Invalid MODE {MODE} for epoch {epch}")
        return accs, {}, 0, 0, (np.array([]), np.array([]))

    if target_test_loss.numel() == 0:
        print(f"Error: Failed to compute target_test_loss for epoch {epch}")
        return accs, {}, 0, 0, (np.array([]), np.array([]))
    print(f"Debug: target_test_loss shape (cos_attack): {target_test_loss.shape}")

    if target_train_loss.dim() == 0:
        target_train_loss = target_train_loss.unsqueeze(0)
    if target_test_loss.dim() == 0:
        target_test_loss = target_test_loss.unsqueeze(0)

    auc, log_auc, tprs = plot_auc("cos", target_test_loss, target_train_loss, epch)
    print(f"__{'_' * 10} cos")
    print(f"Debug: tprs = {tprs}, log_auc = {log_auc}")
    print(f"__{'_' * 10}")

    return accs, tprs, auc, log_auc, (target_test_loss.cpu().numpy(), target_train_loss.cpu().numpy())

def fig_out(save_dir, accs, tprs, auc, log_auc, losses, attack_mode, epoch):
    print(f"Debug: fig_out called with attack_mode={attack_mode}, epoch={epoch}, auc={auc}, log_auc={log_auc}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f'attack_{attack_mode}_epoch{epoch}.png')
    plt.figure()
    plt.plot(losses[0], label='out')
    plt.plot(losses[1], label='in')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def attack_comparison(f, log_path, save_dir, epochs, K, defence, seed):
    print(f"Debug: Inside attack_comparison, f = {f}, type(f) = {type(f)}")
    for epch in epochs:
        print(f"Debug: Processing epoch {epch}")
        for attack_mode in attack_modes:
            print(f"Debug: Running {attack_mode} for epoch {epch}")
            if attack_mode == "cosine attack":
                accs, tprs, auc, log_auc, losses = cos_attack(f, K, epch)
            elif attack_mode == "grad diff":
                accs, tprs, auc, log_auc, losses = lira_attack_ldh_cosine(f, epch, K, save_dir, extract_hinge_loss, "diff")
            elif attack_mode == "loss based":
                accs, tprs, auc, log_auc, losses = lira_attack_ldh_cosine(f, epch, K, save_dir, extract_hinge_loss, "loss")
            elif attack_mode == "grad norm":
                accs, tprs, auc, log_auc, losses = common_attack(f, K, epch)
            else:
                continue

            fig_out(save_dir, accs, tprs, auc, log_auc, losses, attack_mode, epch)
            with open(os.path.join(save_dir, f'attack_log_{attack_mode}.txt'), 'a') as log_file:
                log_file.write(f"Epoch {epch}, AUC: {auc}, Log AUC: {log_auc}, TPRs: {tprs}\n")

def rewrite_print(*args, end=None):
    global SAVE_DIR
    if end is None:
        print(*args)
    else:
        print(*args, end=end)
    try:
        if not SAVE_DIR:
            print("Error: SAVE_DIR is not set.")
            return
        file_path = os.path.join(SAVE_DIR, f'attack_select_{select_mode}_{select_method}_{MODE}_n{SHADOW_NUM}_s{SEED}_running.log')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            if end is None:
                print(*args, file=f)
            else:
                print(*args, end='', file=f)
    except Exception as e:
        print(f"Error writing to log file: {e}")

def main(argv):
    global MODE, attack_modes, PATH, p_folder, device, select_mode, select_method, SHADOW_NUM, SEED, mix_length
    global SAVE_DIR

    attack_modes = ["cosine attack", "grad diff", "loss based", "grad norm"]
    print("Debug: Initialized attack_modes =", attack_modes)

    try:
        epochs = list(range(10, int(argv[2]) + 1, 10))
        print("Debug: epochs =", epochs)
    except IndexError as e:
        print(f"Error: argv[2] not provided. argv = {argv}, Error: {e}")
        sys.exit(1)

    try:
        original_p_folder = argv[1]
        p_folder = os.path.dirname(os.path.dirname(original_p_folder))
        print("Debug: original_p_folder =", original_p_folder)
        print("Debug: p_folder adjusted to =", p_folder)
    except IndexError as e:
        print(f"Error: argv[1] not provided. argv = {argv}, Error: {e}")
        sys.exit(1)

    PATH = original_p_folder
    PATH += "/client_{}_losses_epoch{}.pkl"
    print("Debug: PATH =", PATH)

    print("Debug: Checking .pkl files existence")
    for i in range(5):
        file_path = PATH.format(i, argv[2])
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.")
            sys.exit(1)
        print(f"Debug: File {file_path} exists.")

    try:
        device = argv[3]
        print("Debug: device =", device)
    except IndexError as e:
        print(f"Error: argv[3] not provided. argv = {argv}, Error: {e}")
        sys.exit(1)

    MODE = 'val'
    print("Debug: MODE =", MODE)

    try:
        SEED = int(argv[4])
        print("Debug: SEED =", SEED)
    except IndexError as e:
        print(f"Error: argv[4] not provided. argv = {argv}, Error: {e}")
        sys.exit(1)

    MAX_K = 5
    SAVE_DIR = ""
    print("Debug: MAX_K =", MAX_K)
    print("Debug: SAVE_DIR =", SAVE_DIR)

    print("Debug: Starting os.walk on p_folder =", p_folder)
    for root, dirs, files in os.walk(p_folder, topdown=True):
        print("Debug: root =", root)
        print("Debug: dirs =", dirs)
        print("Debug: files =", files)
        if root == p_folder:
            folder_name = os.path.basename(p_folder)
            try:
                MAX_K = int(folder_name.split("_K")[1].split("_")[0])
                print(f"Debug: Extracted MAX_K = {MAX_K} from folder name {folder_name}")
            except (IndexError, ValueError) as e:
                print(f"Debug: Failed to extract MAX_K from {folder_name}, using default MAX_K = 5, Error: {e}")
        
        for name in dirs:
            full_path = os.path.join(root, name)
            if os.path.exists(os.path.join(full_path, "client_0_losses_epoch1.pkl")):
                model = name.split("_")[3] if "_" in name else "unknown"
                defence = name.split("_")[-5].strip('def').strip('0.0') if len(name.split("_")) > 5 else "none"
                seed = name.split("_")[-1] if "_" in name else "1"
                save_dir = full_path
                SAVE_DIR = save_dir

                folder_name = os.path.basename(p_folder)
                if 'iid1' in folder_name:
                    select_mode = 0
                    select_method = 'none'
                    SHADOW_NUM = 9
                else:
                    select_mode = 1
                    select_method = 'outlier'
                    SHADOW_NUM = 4

                print("Debug: epochs =", epochs)
                print(os.path.join(root, name))
                rewrite_print(os.path.join(root, name))
                print('MODE\tattack_modes\tPATH\tp_folder\tselect_mode\tselect_method\tSHADOW_NUM\tSEED')
                print(f'{MODE}\t{attack_modes}\t{PATH}\t{p_folder}\t{select_mode}\t{select_method}\t{SHADOW_NUM}\t{SEED}')
                print("name:", name)

                if 'cifar100' in name:
                    mix_length = int(10000 / MAX_K)
                elif 'dermnet' in name:
                    mix_length = 4000
                elif 'mnist' in name:
                    mix_length = int(50000 / MAX_K)

                if model == "alexnet":
                    log_path = "logs/log_alex"
                else:
                    log_path = "logs/log_res"

                try:
                    print("Debug: Calling attack_comparison with epochs =", epochs)
                    attack_comparison(PATH, log_path, save_dir, epochs, MAX_K, defence, seed)
                    print("success!")
                except Exception as e:
                    print(f"Error: {e}, MAX_K = {MAX_K}, PATH = {PATH}")
                    sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)