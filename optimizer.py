import torch
import math
from torch.optim import Optimizer

class ScepticalAdam(Optimizer):
    """
    ScepticalAdam: Epistemic Quarantine via Orthogonal Gradient Projection.
    
    This optimizer allows a model to learn from data while mechanically 
    isolating information that does not align with a 'Truth Vector' 
    established during an anchoring phase.
    
    It calculates the Global Cosine Similarity of the entire gradient update 
    against the Truth Vector. If alignment is below the threshold (Slop), 
    it projects the update onto the orthogonal complement of Truth.
    
    Args:
        named_params (iterable): output of model.named_parameters(). 
                                 Used to map parameter IDs to Truth Vector names.
        truth_vectors (dict): Dictionary mapping parameter names to 
                              normalized unit tensors (The Anchor).
        lr (float): Learning rate (default: 1e-3).
        skepticism_threshold (float): Cosine similarity threshold (default: 0.2).
        betas (Tuple[float, float]): Coefficients for running averages (default: (0.9, 0.999)).
        eps (float): Term added to denominator to improve numerical stability.
    """
    def __init__(self, named_params, truth_vectors, lr=1e-3, skepticism_threshold=0.2, 
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        
        # 1. Setup Param Groups (Standard PyTorch)
        # We consume the named_params iterator to get the actual list of parameters
        self.param_map = {id(p): name for name, p in named_params}
        params = [p for name, p in named_params] # Re-create list for super().__init__
        
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(ScepticalAdam, self).__init__(params, defaults)
        
        self.truth_vectors = truth_vectors
        self.skepticism_threshold = skepticism_threshold

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # --- PHASE 1: MEASUREMENT (Global Alignment) ---
        global_dot = 0.0
        grad_norm_sq = 0.0
        truth_norm_sq = 0.0
        
        # We need to iterate ALL parameters first to get the global picture
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Identify if this param has a corresponding Truth Vector
                p_id = id(p)
                if p_id in self.param_map:
                    name = self.param_map[p_id]
                    if name in self.truth_vectors:
                        v_truth = self.truth_vectors[name]
                        
                        # Accumulate Dot Product and Norms
                        global_dot += torch.sum(p.grad * v_truth).item()
                        grad_norm_sq += torch.sum(p.grad ** 2).item()
                        # truth_norm should be 1.0 if normalized, but we check to be safe
                        truth_norm_sq += torch.sum(v_truth ** 2).item()

        grad_norm = math.sqrt(grad_norm_sq)
        truth_norm = math.sqrt(truth_norm_sq)
        
        # Calculate Global Cosine
        if grad_norm > 1e-8 and truth_norm > 1e-8:
            cosine = global_dot / (grad_norm * truth_norm)
        else:
            cosine = 0.0

        # --- PHASE 2: UPDATE (With Quarantine) ---
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # THE INTERVENTION
                # If the GLOBAL update is misaligned (Slop), we project out the Truth component
                # from EVERY parameter individually.
                if abs(cosine) < self.skepticism_threshold:
                    p_id = id(p)
                    if p_id in self.param_map:
                        name = self.param_map[p_id]
                        if name in self.truth_vectors:
                            v_truth = self.truth_vectors[name]
                            
                            # Projection: g' = g - (g . v_total) * v_unit
                            # Note: We use the global alignment scalar for the projection strength
                            projection = (global_dot / (truth_norm_sq + 1e-8)) * v_truth
                            grad.sub_(projection)

                # Standard AdamW Step
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
