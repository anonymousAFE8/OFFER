import torch
from torch.optim import Optimizer
import math


class AdamW_with_LAMB_for_Embeddings(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            apply_lamb = group.get('apply_lamb', False)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('This optimizer does not support sparse gradients.')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if apply_lamb:
      
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    m_hat = exp_avg / bias_correction1
                    v_hat_sqrt = (exp_avg_sq / bias_correction2).sqrt().add_(group['eps'])

                    update = m_hat / v_hat_sqrt
                    update.add_(p.data, alpha=group['weight_decay'])

                    param_norm = torch.norm(p.data)
                    update_norm = torch.norm(update)
                    r = param_norm / update_norm if param_norm > 0 and update_norm > 0 else 1.0
        
                    log_r = torch.log(r)
                        
                    trust_ratio = 1.0 + torch.tanh(log_r/ 100)

                    p.data.add_(update, alpha=-group['lr'] * trust_ratio)

                else:
                    
                    # w_t = w_t - lr * wd * w_t
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                    step_size = group['lr'] / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    
        return loss
