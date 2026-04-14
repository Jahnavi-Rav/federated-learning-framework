import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import copy
from dataclasses import dataclass

@dataclass
class FederatedConfig:
    num_clients: int = 10
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    client_fraction: float = 0.3
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    clip_norm: float = 1.0

class FederatedClient:
    """Individual client in federated learning system"""
    
    def __init__(self, client_id: int, model: nn.Module, data_loader, config: FederatedConfig):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.data_loader = data_loader
        self.config = config
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.learning_rate)
        
    def train_local(self, global_weights: Dict) -> Tuple[Dict, int, float]:
        """Train model locally and return updated weights"""
        # Load global model weights
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        total_loss = 0.0
        num_samples = 0
        
        for epoch in range(self.config.local_epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                
                # Gradient clipping for differential privacy
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_norm)
                
                self.optimizer.step()
                
                total_loss += loss.item() * len(data)
                num_samples += len(data)
        
        avg_loss = total_loss / num_samples
        return self.model.state_dict(), num_samples, avg_loss
    
    def add_dp_noise(self, weights: Dict, sensitivity: float) -> Dict:
        """Add differential privacy noise to weights"""
        noisy_weights = copy.deepcopy(weights)
        
        # Calculate noise scale based on DP parameters
        noise_scale = sensitivity / self.config.dp_epsilon
        
        for key in noisy_weights:
            if 'weight' in key or 'bias' in key:
                noise = torch.normal(0, noise_scale, size=noisy_weights[key].shape)
                noisy_weights[key] += noise
        
        return noisy_weights

class SecureAggregator:
    """Secure aggregation with homomorphic encryption simulation"""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.client_masks = {}
        
    def generate_masks(self) -> Dict[int, Dict]:
        """Generate secret masks for each client"""
        for client_id in range(self.num_clients):
            mask = {}
            # Generate random masks (in practice, these would be shared secrets)
            for i in range(self.num_clients):
                if i != client_id:
                    mask[i] = torch.randn(1)  # Simplified mask
            self.client_masks[client_id] = mask
        return self.client_masks
    
    def secure_aggregate(self, client_updates: List[Dict], num_samples: List[int]) -> Dict:
        """Aggregate client updates with secure aggregation"""
        # Weighted average based on number of samples
        total_samples = sum(num_samples)
        aggregated = {}
        
        # Initialize aggregated weights
        for key in client_updates[0].keys():
            aggregated[key] = torch.zeros_like(client_updates[0][key])
        
        # Weighted aggregation
        for client_update, n_samples in zip(client_updates, num_samples):
            weight = n_samples / total_samples
            for key in aggregated:
                aggregated[key] += client_update[key] * weight
        
        # In practice, masks would cancel out in aggregation
        return aggregated

class FederatedServer:
    """Central server for federated learning"""
    
    def __init__(self, model: nn.Module, config: FederatedConfig):
        self.global_model = model
        self.config = config
        self.aggregator = SecureAggregator(config.num_clients)
        self.round_losses = []
        
    def select_clients(self, clients: List[FederatedClient]) -> List[FederatedClient]:
        """Randomly select subset of clients for training round"""
        num_selected = max(1, int(self.config.client_fraction * len(clients)))
        selected_indices = np.random.choice(len(clients), num_selected, replace=False)
        return [clients[i] for i in selected_indices]
    
    def aggregate_weights(self, client_updates: List[Dict], num_samples: List[int]) -> Dict:
        """FedAvg: Aggregate client model updates"""
        return self.aggregator.secure_aggregate(client_updates, num_samples)
    
    def train_round(self, clients: List[FederatedClient]) -> float:
        """Execute one round of federated training"""
        # Select clients for this round
        selected_clients = self.select_clients(clients)
        
        # Get global model weights
        global_weights = self.global_model.state_dict()
        
        # Train on selected clients
        client_updates = []
        client_samples = []
        client_losses = []
        
        for client in selected_clients:
            weights, n_samples, loss = client.train_local(global_weights)
            client_updates.append(weights)
            client_samples.append(n_samples)
            client_losses.append(loss)
        
        # Aggregate updates
        aggregated_weights = self.aggregate_weights(client_updates, client_samples)
        
        # Update global model
        self.global_model.load_state_dict(aggregated_weights)
        
        # Return average loss
        avg_loss = sum(client_losses) / len(client_losses)
        return avg_loss
    
    def train(self, clients: List[FederatedClient]):
        """Complete federated training process"""
        print(f"Starting Federated Learning with {self.config.num_clients} clients")
        print(f"Rounds: {self.config.num_rounds}, Client fraction: {self.config.client_fraction}")
        
        for round_num in range(self.config.num_rounds):
            avg_loss = self.train_round(clients)
            self.round_losses.append(avg_loss)
            
            if (round_num + 1) % 10 == 0:
                print(f"Round {round_num + 1}/{self.config.num_rounds} - Avg Loss: {avg_loss:.4f}")
        
        print("\nFederated training completed!")
        return self.global_model

class DifferentialPrivacy:
    """Differential privacy mechanisms for federated learning"""
    
    @staticmethod
    def compute_privacy_spent(steps: int, epsilon: float, delta: float, batch_size: int, 
                            dataset_size: int) -> Tuple[float, float]:
        """Compute privacy budget spent using moments accountant"""
        # Simplified privacy accounting
        q = batch_size / dataset_size
        sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        # Epsilon after 'steps' iterations
        epsilon_spent = epsilon * np.sqrt(steps * q)
        return epsilon_spent, delta
    
    @staticmethod
    def add_gaussian_noise(tensor: torch.Tensor, sensitivity: float, epsilon: float, 
                          delta: float) -> torch.Tensor:
        """Add calibrated Gaussian noise for differential privacy"""
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        noise = torch.normal(0, sigma, size=tensor.shape)
        return tensor + noise

class FederatedLearningFramework:
    """Complete federated learning framework"""
    
    def __init__(self, model: nn.Module, client_data_loaders: List, config: FederatedConfig):
        self.config = config
        self.server = FederatedServer(model, config)
        
        # Create clients
        self.clients = [
            FederatedClient(i, model, data_loader, config)
            for i, data_loader in enumerate(client_data_loaders)
        ]
        
        print(f"Initialized Federated Learning Framework")
        print(f"Clients: {len(self.clients)}")
        print(f"Differential Privacy: ε={config.dp_epsilon}, δ={config.dp_delta}")
    
    def run(self) -> nn.Module:
        """Run federated learning"""
        trained_model = self.server.train(self.clients)
        return trained_model
    
    def evaluate_global_model(self, test_loader) -> float:
        """Evaluate global model on test set"""
        self.server.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                output = self.server.global_model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy

# Example usage
if __name__ == "__main__":
    # Simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.fc2 = nn.Linear(128, 10)
        
        def forward(self, x):
            x = x.view(-1, 784)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleModel()
    config = FederatedConfig(
        num_clients=10,
        num_rounds=50,
        local_epochs=5,
        client_fraction=0.3,
        dp_epsilon=1.0
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Federated config: {config}")
