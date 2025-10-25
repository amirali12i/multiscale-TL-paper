"""
Hessian Analysis for Neural Networks

Computes Hessian trace and condition number using three independent methods:
1. Finite differences
2. Automatic differentiation (exact)
3. Hutchinson's stochastic estimator

Example usage:
    python hessian_analysis.py \
        --checkpoint ./outputs/bert_mnli/checkpoint-best.pt \
        --data mnli \
        --methods all
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModel ForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HessianAnalyzer:
    """
    Compute Hessian properties of neural networks
    
    Methods:
        - finite_differences: Use finite differences (approximate but intuitive)
        - autograd: Use PyTorch's automatic differentiation (exact)
        - hutchinson: Use Hutchinson's stochastic trace estimator (scalable)
    """
    
    def __init__(self, model, loss_fn, device='cuda'):
        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.model.to(device)
    
    def compute_trace_finite_differences(
        self,
        dataloader,
        epsilon=1e-4,
        num_samples=1000,
        num_directions=100
    ):
        """
        Compute Hessian trace using finite differences
        
        tr(H) ≈ sum_i [L(θ + ε*e_i) - 2*L(θ) + L(θ - ε*e_i)] / ε²
        
        Args:
            dataloader: Data for computing loss
            epsilon: Perturbation size
            num_samples: Number of data samples to use
            num_directions: Number of random directions to sample
        
        Returns:
            Estimated Hessian trace
        """
        logger.info("Computing Hessian trace via finite differences...")
        
        # Get parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Compute base loss
        base_loss = self._compute_loss(dataloader, num_samples)
        
        # Sample random directions
        trace_estimate = 0.0
        
        for _ in tqdm(range(num_directions), desc="Sampling directions"):
            # Random direction (Rademacher)
            direction = [torch.randint_like(p, 0, 2) * 2.0 - 1.0 for p in params]
            
            # Normalize direction
            norm = torch.sqrt(sum((d ** 2).sum() for d in direction))
            direction = [d / norm for d in direction]
            
            # Perturb parameters: θ + ε*d
            with torch.no_grad():
                for p, d in zip(params, direction):
                    p.add_(epsilon * d)
            
            loss_plus = self._compute_loss(dataloader, num_samples)
            
            # Restore and perturb: θ - ε*d
            with torch.no_grad():
                for p, d in zip(params, direction):
                    p.sub_(2 * epsilon * d)
            
            loss_minus = self._compute_loss(dataloader, num_samples)
            
            # Restore original parameters
            with torch.no_grad():
                for p, d in zip(params, direction):
                    p.add_(epsilon * d)
            
            # Second-order finite difference
            second_derivative = (loss_plus - 2 * base_loss + loss_minus) / (epsilon ** 2)
            trace_estimate += second_derivative
        
        # Average over directions
        trace_estimate /= num_directions
        
        # Scale by total number of parameters
        total_params = sum(p.numel() for p in params)
        trace_estimate *= total_params
        
        return float(trace_estimate)
    
    def compute_trace_autograd(self, dataloader, num_samples=1000):
        """
        Compute exact Hessian trace using automatic differentiation
        
        tr(H) = sum_i d²L/dθ_i²
        
        Args:
            dataloader: Data for computing loss
            num_samples: Number of data samples to use
        
        Returns:
            Exact Hessian trace
        """
        logger.info("Computing Hessian trace via autograd...")
        
        # Get parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Compute loss with create_graph=True
        loss = self._compute_loss(dataloader, num_samples, create_graph=True)
        
        # First derivatives
        grads = torch.autograd.grad(
            loss, params,
            create_graph=True,
            retain_graph=True
        )
        
        # Compute trace of Hessian (sum of diagonal elements)
        trace = 0.0
        
        for grad, param in tqdm(zip(grads, params), 
                               total=len(params),
                               desc="Computing diagonal"):
            # For efficiency, compute trace contribution from this parameter
            # tr(H_param) = sum_i d²L/dθ_i²
            grad_norm = (grad * grad).sum()
            
            # Second derivative
            hess_diag = torch.autograd.grad(
                grad_norm, param,
                retain_graph=True,
                create_graph=False
            )[0]
            
            trace += hess_diag.sum().item()
        
        return float(trace)
    
    def compute_trace_hutchinson(
        self,
        dataloader,
        num_samples=1000,
        num_vectors=100
    ):
        """
        Compute Hessian trace using Hutchinson's stochastic estimator
        
        tr(H) ≈ (1/m) * sum_j v_j^T H v_j
        where v_j are random Rademacher vectors
        
        Args:
            dataloader: Data for computing loss
            num_samples: Number of data samples to use
            num_vectors: Number of random vectors for estimation
        
        Returns:
            Estimated Hessian trace
        """
        logger.info("Computing Hessian trace via Hutchinson estimator...")
        
        # Get parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        trace_estimate = 0.0
        
        for _ in tqdm(range(num_vectors), desc="Sampling vectors"):
            # Random Rademacher vector
            v = [torch.randint_like(p, 0, 2) * 2.0 - 1.0 for p in params]
            
            # Compute loss
            loss = self._compute_loss(dataloader, num_samples, create_graph=True)
            
            # Compute gradient
            grads = torch.autograd.grad(
                loss, params,
                create_graph=True,
                retain_graph=True
            )
            
            # Compute v^T * grad
            grad_v = sum((g * vec).sum() for g, vec in zip(grads, v))
            
            # Compute Hessian-vector product: H*v
            hvp = torch.autograd.grad(
                grad_v, params,
                retain_graph=False,
                create_graph=False
            )
            
            # Compute v^T * H * v
            trace_estimate += sum(
                (h * vec).sum().item() for h, vec in zip(hvp, v)
            )
        
        # Average over random vectors
        trace_estimate /= num_vectors
        
        return float(trace_estimate)
    
    def compute_condition_number(
        self,
        dataloader,
        num_samples=1000,
        num_iterations=10
    ):
        """
        Estimate condition number using power iteration
        
        κ(H) = λ_max / λ_min
        
        Args:
            dataloader: Data for computing loss
            num_samples: Number of data samples
            num_iterations: Number of power iterations
        
        Returns:
            Estimated condition number
        """
        logger.info("Estimating condition number...")
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Estimate largest eigenvalue (power iteration)
        v_max = [torch.randn_like(p) for p in params]
        norm = torch.sqrt(sum((v ** 2).sum() for v in v_max))
        v_max = [v / norm for v in v_max]
        
        for _ in range(num_iterations):
            # Compute H*v
            loss = self._compute_loss(dataloader, num_samples, create_graph=True)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            grad_v = sum((g * v).sum() for g, v in zip(grads, v_max))
            hvp = torch.autograd.grad(grad_v, params)
            
            # Normalize
            v_max = list(hvp)
            norm = torch.sqrt(sum((v ** 2).sum() for v in v_max))
            v_max = [v / norm for v in v_max]
        
        # Compute Rayleigh quotient for λ_max
        loss = self._compute_loss(dataloader, num_samples, create_graph=True)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_v = sum((g * v).sum() for g, v in zip(grads, v_max))
        hvp = torch.autograd.grad(grad_v, params)
        lambda_max = sum((h * v).sum().item() for h, v in zip(hvp, v_max))
        
        # Estimate smallest eigenvalue (inverse power iteration)
        v_min = [torch.randn_like(p) for p in params]
        norm = torch.sqrt(sum((v ** 2).sum() for v in v_min))
        v_min = [v / norm for v in v_min]
        
        # Use shifted inverse power iteration
        # (H - σI)^{-1} * v ≈ v - (H*v) / ||H*v||
        shift = lambda_max * 0.1  # Small shift
        
        for _ in range(num_iterations):
            loss = self._compute_loss(dataloader, num_samples, create_graph=True)
            grads = torch.autograd.grad(loss, params, create_graph=True)
            grad_v = sum((g * v).sum() for g, v in zip(grads, v_min))
            hvp = torch.autograd.grad(grad_v, params)
            
            # Shifted inverse: v - (H*v - shift*v)
            v_min = [v - (h - shift * v) for v, h in zip(v_min, hvp)]
            norm = torch.sqrt(sum((v ** 2).sum() for v in v_min))
            v_min = [v / norm for v in v_min]
        
        # Compute Rayleigh quotient for λ_min
        loss = self._compute_loss(dataloader, num_samples, create_graph=True)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_v = sum((g * v).sum() for g, v in zip(grads, v_min))
        hvp = torch.autograd.grad(grad_v, params)
        lambda_min = sum((h * v).sum().item() for h, v in zip(hvp, v_min))
        
        # Condition number
        condition_number = abs(lambda_max / (lambda_min + 1e-10))
        
        return float(condition_number), float(lambda_max), float(lambda_min)
    
    def _compute_loss(self, dataloader, num_samples, create_graph=False):
        """Compute average loss over data samples"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.set_grad_enabled(create_graph):
            for batch in dataloader:
                if num_batches * batch['input_ids'].size(0) >= num_samples:
                    break
                
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                total_loss += loss
                num_batches += 1
        
        return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='mnli',
                       help='GLUE task name')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['finite_diff', 'autograd', 'hutchinson'],
                       help='Hessian computation methods')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of data samples')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='hessian_results.json')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.checkpoint}")
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    
    # Load data
    logger.info(f"Loading {args.data} dataset")
    dataset = load_dataset('glue', args.data, split='validation')
    
    # Create dataloader
    def collate_fn(examples):
        return tokenizer.pad(
            [tokenizer(ex['sentence'], truncation=True, max_length=128) 
             for ex in examples],
            return_tensors='pt'
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn
    )
    
    # Initialize analyzer
    def loss_fn(model, batch):
        outputs = model(**batch)
        return outputs.loss
    
    analyzer = HessianAnalyzer(model, loss_fn, args.device)
    
    # Compute Hessian properties
    results = {}
    
    if 'all' in args.methods or 'finite_diff' in args.methods:
        trace_fd = analyzer.compute_trace_finite_differences(
            dataloader, num_samples=args.num_samples
        )
        results['trace_finite_diff'] = trace_fd
        logger.info(f"Hessian Trace (Finite Diff): {trace_fd:.4e}")
    
    if 'all' in args.methods or 'autograd' in args.methods:
        trace_autograd = analyzer.compute_trace_autograd(
            dataloader, num_samples=args.num_samples
        )
        results['trace_autograd'] = trace_autograd
        logger.info(f"Hessian Trace (Autograd): {trace_autograd:.4e}")
    
    if 'all' in args.methods or 'hutchinson' in args.methods:
        trace_hutchinson = analyzer.compute_trace_hutchinson(
            dataloader, num_samples=args.num_samples
        )
        results['trace_hutchinson'] = trace_hutchinson
        logger.info(f"Hessian Trace (Hutchinson): {trace_hutchinson:.4e}")
    
    # Compute statistics if multiple methods used
    traces = [v for k, v in results.items() if k.startswith('trace_')]
    if len(traces) > 1:
        results['trace_mean'] = np.mean(traces)
        results['trace_std'] = np.std(traces)
        results['trace_relative_std'] = np.std(traces) / np.mean(traces)
        logger.info(f"Mean: {results['trace_mean']:.4e} ± {results['trace_std']:.4e}")
        logger.info(f"Relative Std: {results['trace_relative_std']:.2%}")
    
    # Compute condition number
    if 'condition' in args.methods or 'all' in args.methods:
        cond, lambda_max, lambda_min = analyzer.compute_condition_number(
            dataloader, num_samples=args.num_samples
        )
        results['condition_number'] = cond
        results['lambda_max'] = lambda_max
        results['lambda_min'] = lambda_min
        logger.info(f"Condition Number: {cond:.4e}")
        logger.info(f"λ_max: {lambda_max:.4e}, λ_min: {lambda_min:.4e}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()
