import torch
import math
from torch.optim import Optimizer

class ProjectedSkepticalAdam(Optimizer):
    """
    Implements Epistemic Quarantine via Orthogonal Gradient Projection.
    
    This optimizer allows a model to learn from data while mechanically 
    isolating information that does not align with a 'Truth Vector' 
    established during an anchoring phase.
    
    Args:
        params (iterable): Iterable of parameters to optimize.
        lr (float): Learning rate (default: 1e-5).
        truth_units (dict): Dictionary mapping parameter names to 
            normalized unit tensors from the Anchor phase.
        threshold (float): Cosine similarity threshold below which 
            orthogonal projection is triggered (default: 0.15).
        betas (Tuple[float, float]): Coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): Term added to denominator to improve numerical stability.
    """
    def __init__(self, params, lr=1e-5, truth_units=None, threshold=0.15, 
                 betas=(0.9, 0.999), eps=1e-8):
        if truth_units is None:
            raise ValueError("ProjectedSkepticalAdam requires a truth_units anchor.")
            
        defaults = dict(lr=lr, threshold=threshold, betas=betas, eps=eps)
        super(ProjectedSkepticalAdam, self).__init__(params, defaults)
        self.truth_units = truth_units

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1. CALCULATE GLOBAL ALIGNMENT (Epistemic Switch)
        # We perform a global pass to see if this batch is 'Slop' or 'Truth'
        global_dot = 0.0
        grad_norm_sq = 0.0
        
        # Mapping params to truth_units keys (assumes params are named in the same order)
        # In a production environment, you would ensure keys match model.named_parameters()
        unit_list = list(self.truth_units.values())
        param_list = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_list.append(p)

        for p, v_unit in zip(param_list, unit_list):
            global_dot += torch.sum(p.grad * v_unit).item()
            grad_norm_sq += torch.sum(p.grad ** 2).item()

        grad_norm = math.sqrt(grad_norm_sq)
        cosine = global_dot / (grad_norm + 1e-8)

        # 2. APPLY UPDATES
        for group in self.param_groups:
            threshold = group['threshold']
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p, v_unit in zip(param_list, unit_list):
                grad = p.grad
                
                # THE QUARANTINE: If alignment is low, strip the truth component
                # This ensures the update is purely orthogonal to the anchor logic
                if abs(cosine) < threshold:
                    # g' = g - (g . v_unit) * v_unit
                    # Note: global_dot is used here as the projection scalar
                    projection = global_dot * v_unit
                    grad.sub_(projection)

                # Standard Adam-like update logic on the (possibly projected) gradient
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
