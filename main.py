import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from dataset import MNIST, CIFAR100
from models import AlexNet
import time
import pickle
import argparse
import torch.nn.functional as F

def setup_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def split_data(args):
    """Split dataset into train, validation, and test indices."""
    if args.dataset == 'mnist':
        total_train_samples = 60000
        test_samples = 10000
        val_samples = 5000
    else:  # CIFAR100
        total_train_samples = 50000
        test_samples = 10000
        val_samples = 5000

    train_samples = total_train_samples - val_samples
    train_indices = np.arange(total_train_samples)
    test_indices = np.arange(test_samples)
    np.random.shuffle(train_indices)
    train_indices, val_indices = train_indices[:train_samples], train_indices[train_samples:]

    if args.iid:
        user_indices = np.array_split(train_indices, args.num_users)
    else:
        user_indices = [[] for _ in range(args.num_users)]
        for idx in train_indices:
            dataset = MNIST(args.data_root, [idx], train=True, download=True, need_index=True)
            label = dataset.targets[0]
            user_id = np.random.choice(args.num_users, p=np.random.dirichlet([args.beta] * args.num_users))
            user_indices[user_id].append(idx)

    return user_indices, val_indices, test_indices

def get_dataset(args, train_indices, test_indices, val_indices):
    """Load datasets with appropriate transforms."""
    if args.dataset == 'mnist':
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:  # CIFAR100
        num_classes = 100
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

    train_dataset = MNIST(args.data_root, train_indices, train=True, transform=train_transform, download=True, need_index=True)
    val_dataset = MNIST(args.data_root, val_indices, train=True, transform=test_transform, download=True, need_index=True)
    test_dataset = MNIST(args.data_root, test_indices, train=False, transform=test_transform, download=True, need_index=True)

    return train_dataset, test_dataset, val_dataset, num_classes

def compute_entropy(logits):
    """Compute entropy of logits (per-sample)."""
    probs = torch.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    return entropy  # Trả về tensor 1D [batch_size]

def compute_cosine_similarity_per_sample(model, data, device):
    """Compute cosine similarity for each sample in the batch."""
    cos_sims = []
    for i in range(data.size(0)):  # Lặp qua từng mẫu trong batch
        model.zero_grad()
        single_input = data[i:i+1].to(device)  # Lấy mẫu thứ i
        output = model(single_input)
        output.backward(torch.ones_like(output))  # Gradient giả để tính
        grads = [p.grad.clone().detach().cpu() for p in model.parameters() if p.grad is not None]
        params = [p.detach().cpu() for p in model.parameters()]
        
        # Tính cosine similarity cho mẫu này
        cos_sim_sum = 0.0
        count = 0
        for g, p in zip(grads, params):
            if g is None or p is None:
                continue
            g_flat = g.flatten()
            p_flat = p.flatten()
            cos_sim = torch.nn.functional.cosine_similarity(g_flat, p_flat, dim=0)
            cos_sim_sum += cos_sim.item()
            count += 1
        avg_cos_sim = cos_sim_sum / count if count > 0 else 0.0
        cos_sims.append(avg_cos_sim)
    
    return torch.tensor(cos_sims, device='cpu')

def compute_grad_norm_per_sample(model, data, device):
    """Compute gradient norm for each sample in the batch."""
    grad_norms = []
    for i in range(data.size(0)):
        model.zero_grad()
        single_input = data[i:i+1].to(device)
        output = model(single_input)
        output.backward(torch.ones_like(output))
        grads = [p.grad.clone().detach().cpu() for p in model.parameters() if p.grad is not None]
        grad_norm = sum(torch.norm(g).item()**2 for g in grads if g is not None)**0.5
        grad_norms.append(grad_norm)
    
    return torch.tensor(grad_norms, device='cpu')

def test(model, dataloader, criterion, device):
    """Evaluate model on a dataset and compute per-sample metrics."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_logits = []
    all_labels = []
    all_entropies = []
    all_grad_norms = []
    all_diffs = []
    all_cos_sims = []

    for batch_idx, (data, target, _) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # Tính output và loss
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Lưu logits và labels
        all_logits.append(output.detach().cpu())
        all_labels.append(target.detach().cpu())

        # Tính entropy cho batch
        batch_entropy = compute_entropy(output.detach().cpu())
        all_entropies.append(batch_entropy)

        # Tính cosine similarity và grad norm cho từng mẫu
        batch_cos_sims = compute_cosine_similarity_per_sample(model, data, device)
        all_cos_sims.append(batch_cos_sims)

        batch_grad_norms = compute_grad_norm_per_sample(model, data, device)
        all_grad_norms.append(batch_grad_norms)

        # Tính diffs (dựa trên grad_norm giữa các batch)
        if batch_idx > 0:
            prev_grad_norms = all_grad_norms[-2]
            batch_diffs = torch.abs(batch_grad_norms - prev_grad_norms.mean())
            all_diffs.append(batch_diffs)

    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total

    # Ghép các giá trị thành mảng
    test_cos = torch.cat(all_cos_sims, dim=0)  # [num_samples]
    test_grad_norm = torch.cat(all_grad_norms, dim=0)  # [num_samples]
    test_diffs = torch.cat(all_diffs, dim=0) if all_diffs else torch.zeros_like(test_cos)  # [num_samples]
    test_entropy = torch.cat(all_entropies, dim=0)  # [num_samples]
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return avg_loss, accuracy, {
        'logit': logits,
        'labels': labels,
        'cos': test_cos,
        'grad_norm': test_grad_norm,
        'diffs': test_diffs,
        'entropy': test_entropy
    }

def main(args):
    """Main function for federated learning."""
    setup_seed(args.seed)
    device = torch.device(args.device)

    # Create save_dir based on IID or non-IID
    args.save_dir = os.path.join(args.save_dir, f"{args.dataset}_K{args.num_users}_N{args.samples_per_user}_{args.model_name}_s{args.seed}_iid{args.iid}")
    args.log_folder_name = args.save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print("Scores saved in:", os.path.join(os.getcwd(), args.save_dir))

    # Split data
    user_indices, val_indices, test_indices = split_data(args)

    # Load datasets
    train_dataset, test_dataset, val_dataset, num_classes = get_dataset(args, user_indices[0], test_indices, val_indices)

    # Create DataLoaders with balanced sampling
    train_loaders = []
    for i in range(args.num_users):
        client_dataset = MNIST(args.data_root, user_indices[i], train=True, transform=train_dataset.transform, download=True, need_index=True)
        # Đồng bộ số mẫu
        if len(client_dataset) > args.samples_per_user:
            client_dataset, _ = torch.utils.data.random_split(client_dataset, [args.samples_per_user, len(client_dataset) - args.samples_per_user])
        train_loaders.append(torch.utils.data.DataLoader(client_dataset, batch_size=args.batch_size, shuffle=True))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize global model
    global_model = AlexNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    # Epoch loop
    for epoch in range(1, args.epochs + 1):
        # Train each client
        client_models = []
        train_res_list = []
        for client_idx in range(args.num_users):
            # Initialize client model and optimizer
            client_model = AlexNet(num_classes=num_classes).to(device)
            client_model.load_state_dict(global_model.state_dict())
            optimizer = optim.SGD(client_model.parameters(), lr=args.lr)

            # Train client
            client_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            all_logits = []
            all_labels = []
            all_entropies = []
            all_grad_norms = []
            all_diffs = []
            all_cos_sims = []

            for batch_idx, (data, target, _) in enumerate(train_loaders[client_idx]):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = client_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                # Tính các chỉ số per-sample
                batch_cos_sims = compute_cosine_similarity_per_sample(client_model, data, device)
                all_cos_sims.append(batch_cos_sims)

                batch_grad_norms = compute_grad_norm_per_sample(client_model, data, device)
                all_grad_norms.append(batch_grad_norms)

                # Tính diffs (dựa trên grad_norm giữa các batch)
                if batch_idx > 0:
                    prev_grad_norms = all_grad_norms[-2]
                    batch_diffs = torch.abs(batch_grad_norms - prev_grad_norms.mean())
                    all_diffs.append(batch_diffs)

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Lưu logits, labels, entropy
                all_logits.append(output.detach().cpu())
                all_labels.append(target.detach().cpu())
                batch_entropy = compute_entropy(output.detach().cpu())
                all_entropies.append(batch_entropy)

            train_loss = running_loss / len(train_loaders[client_idx])
            train_acc = 100. * correct / total

            # Ghép các giá trị thành mảng
            train_cos = torch.cat(all_cos_sims, dim=0)  # [num_samples]
            train_grad_norm = torch.cat(all_grad_norms, dim=0)  # [num_samples]
            train_diffs = torch.cat(all_diffs, dim=0) if all_diffs else torch.zeros_like(train_cos)  # [num_samples]
            train_entropy = torch.cat(all_entropies, dim=0)  # [num_samples]
            logits = torch.cat(all_logits, dim=0)
            labels = torch.cat(all_labels, dim=0)

            # Lưu kết quả train của client
            train_res = {
                'logit': logits,
                'labels': labels,
                'loss': train_loss,
                'acc': train_acc,
                'cos': train_cos,
                'diffs': train_diffs,
                'grad_norm': train_grad_norm,
                'entropy': train_entropy
            }
            train_res_list.append(train_res)
            client_models.append(client_model.state_dict())

        # Aggregate client models (Federated Averaging)
        global_dict = global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([client_models[i][k].float() for i in range(args.num_users)], 0).mean(0)
        global_model.load_state_dict(global_dict)

        # Evaluate on validation and test sets
        val_loss, val_acc, val_res = test(global_model, val_loader, criterion, device)
        test_loss, test_acc, test_res = test(global_model, test_loader, criterion, device)
        print(f'Epoch {epoch}/{args.epochs}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

        # Save results for each client with correct directory
        save_dir = os.path.join(args.save_dir, 'alexnet', 'mnist')
        os.makedirs(save_dir, exist_ok=True)
        for client_id in range(args.num_users):
            save_data = {
                'train_res': train_res_list[client_id],
                'val_res': val_res,
                'test_res': test_res,
                'train_index': user_indices[client_id],
                'val_index': val_indices,
                'test_index': test_indices,
                'test_acc': test_acc,
                'train_cos': train_res_list[client_id]['cos'],
                'train_diffs': train_res_list[client_id]['diffs'],
                'train_grad_norm': train_res_list[client_id]['grad_norm'],
                'train_entropy': train_res_list[client_id]['entropy'],
                'val_cos': val_res.get('cos', torch.tensor(0.0)),
                'val_grad_norm': val_res.get('grad_norm', torch.tensor(0.0)),
                'val_diffs': val_res.get('diffs', torch.tensor(0.0)),
                'val_entropy': val_res.get('entropy', torch.tensor(0.0)),
                'test_cos': test_res.get('cos', torch.tensor(0.0)),
                'test_grad_norm': test_res.get('grad_norm', torch.tensor(0.0)),
                'test_diffs': test_res.get('diffs', torch.tensor(0.0)),
                'test_entropy': test_res.get('entropy', torch.tensor(0.0))
            }
            torch.save(save_data, os.path.join(save_dir, f'client_{client_id}_losses_epoch{epoch}.pkl'))

if __name__ == '__main__':
    def parser_args():
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='mnist')
        parser.add_argument('--data_root', type=str, default='./data')
        parser.add_argument('--save_dir', type=str, default='log_fedmia/noniid')
        parser.add_argument('--log_folder_name', type=str, default='log_fedmia/noniid')
        parser.add_argument('--model_name', type=str, default='alexnet')
        parser.add_argument('--num_users', type=int, default=5)
        parser.add_argument('--samples_per_user', type=int, default=10000)
        parser.add_argument('--epochs', type=int, default=150)
        parser.add_argument('--local_ep', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--lr_up', type=str, default='cosine')
        parser.add_argument('--optim', type=str, default='sgd')
        parser.add_argument('--device', type=str, default='cpu')
        parser.add_argument('--seed', type=int, default=1)
        parser.add_argument('--iid', type=int, default=0)
        parser.add_argument('--beta', type=float, default=1.0)
        parser.add_argument('--MIA_mode', type=int, default=1)
        return parser.parse_args()

    args = parser_args()
    print(args)
    setup_seed(args.seed)
    main(args)
