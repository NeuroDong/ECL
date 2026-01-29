import torch
import torch.nn as nn
import math
import numpy as np

EPS = 1e-5

def is_softmax_output(vec, atol=1e-6):
    """
    vec: 1D tensor of shape (C,) or 2D tensor (N, C)
    """
    if vec.dim() == 1:
        s = vec.sum().item()
        in_range = torch.all((vec >= -atol) & (vec <= 1 + atol)).item()
        return in_range and abs(s - 1.0) <= atol
    elif vec.dim() == 2:
        s = vec.sum(dim=1)
        in_range = torch.all((vec >= -atol) & (vec <= 1 + atol)).item()
        return in_range and torch.all(torch.abs(s - 1.0) <= atol).item()
    else:
        raise ValueError("Expect 1D or 2D tensor.")

def Check_data(logits, labels = None):
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    if labels is not None:
        if not isinstance(labels, torch.Tensor):
            labels = torch.as_tensor(labels)
    if len(logits.shape) == 2:
        if not is_softmax_output(logits):
            probs = torch.softmax(logits, dim=1)
        else:
            probs = logits
    else:
        raise ValueError("Expect 2D tensor.")
    return probs, labels

def soft_binning_ece(
        predictions,
        labels,
        soft_binning_bins = 15,
        soft_binning_use_decay=True,
        soft_binning_decay_factor=0.9,
        soft_binning_temp=0.01
):
    """Computes and returns the soft-binned ECE (binned) scalar tensor (PyTorch).

    Soft-binned ECE (binned, L2-norm) is defined in Eq. (11) of:
    https://arxiv.org/abs/2108.00106. This is a softened version of ECE (binned)
    defined in Eq. (6).

    Args:
        predictions: 1D tensor (N,) or (N,C)
        labels: 1D tensor (N,) of incorrect(0)/correct(1) labels.
        soft_binning_bins: number of bins (int).
        soft_binning_use_decay: whether temp is determined by decay factor (bool).
        soft_binning_decay_factor: approximate decay factor between successive bins.
        soft_binning_temp: soft binning temperature (float) when not using decay.

    Returns:
        A 0-dim torch.Tensor containing the soft-binned ECE value.
    """

    # Convert inputs to torch tensors (float) and ensure 1D shape
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.as_tensor(predictions)
    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)
    if len(predictions.shape) == 2:
        if is_softmax_output(predictions):
            predictions = predictions.max(dim=1).values
        else:
            predictions = torch.softmax(predictions, dim=1).max(dim=1).values
    

    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    if predictions.numel() != labels.numel():
        raise ValueError("predictions and labels must have the same number of elements")

    # Dtype/device handling
    if not predictions.is_floating_point():
        predictions = predictions.to(dtype=torch.get_default_dtype())
    dtype = predictions.dtype
    device = predictions.device
    labels = labels.to(dtype=dtype, device=device)

    B = int(soft_binning_bins)
    if B <= 0:
        raise ValueError("soft_binning_bins must be a positive integer")

    # Bin anchors: midpoints (2i+1)/(2B) for i=0..B-1
    anchors = (torch.arange(B, dtype=dtype, device=device) * 2 + 1) / (2.0 * B)

    # Temperature via decay if requested
    if soft_binning_use_decay:
        soft_binning_temp = -1.0 / (math.log(soft_binning_decay_factor) * B * B)

    # Soft assignment to bins
    diffs = predictions.unsqueeze(1) - anchors.unsqueeze(0)  # [N, B]
    scores = - (diffs ** 2) / soft_binning_temp              # [N, B]
    coeffs = torch.softmax(scores, dim=1)                    # [N, B]

    # Aggregate per-bin statistics
    sum_coeffs_for_bin = coeffs.sum(dim=0)                   # [B]
    denom = torch.clamp(sum_coeffs_for_bin, min=EPS)

    net_bin_confidence = (predictions.unsqueeze(1) * coeffs).sum(dim=0) / denom
    net_bin_accuracy = (labels.unsqueeze(1) * coeffs).sum(dim=0) / denom

    # Bin weights: L1-normalized sum of coeffs per bin
    total = torch.clamp(sum_coeffs_for_bin.sum(), min=EPS)
    bin_weights = sum_coeffs_for_bin / total                  # [B]

    # ECE: sqrt( sum_b w_b * (conf_b - acc_b)^2 )
    ece = torch.sqrt(((net_bin_confidence - net_bin_accuracy) ** 2 * bin_weights).sum())

    return ece

def get_simplex_anchors(num_bins, num_classes):
    """
    Generate anchors (grid points) on the (num_classes-1)-simplex.
    Tries to find a grid resolution N such that total points approx num_bins.
    For K=3, num_bins=15 => N=4 (exact).
    """
    # Simple search for N
    N = 1
    while True:
        # Number of points in regular grid on simplex
        # (N+K-1) choose (K-1)
        count = math.comb(N + num_classes - 1, num_classes - 1)
        if count >= num_bins:
            break
        N += 1
        if N > 50: break 
    
    # Generate grid recursively
    def recursive_gen(n, dim):
        if dim == 1:
            return [[n]]
        res = []
        for i in range(n + 1):
            sub_res = recursive_gen(n - i, dim - 1)
            for sub in sub_res:
                res.append([i] + sub)
        return res

    grid = recursive_gen(N, num_classes) # List of lists (ints)
    # Use centroids (shifted points) to avoid boundary vertices, ensuring anchors are strictly inside the simplex
    anchors = (torch.tensor(grid, dtype=torch.float32) + 1.0 / num_classes) / (float(N) + 1.0)
    return anchors

class ECLossMiniBatch(nn.Module):
    """
    Mini-batch trainable version of In-training ECLoss based on Algorithm 1.
    Implemented as a class to manage stateful auxiliary variables internally.
    """
    def __init__(self,
                 calibration_paradigm,
                 num_bins: int = 15,
                 num_classes: int = 3,
                 lambda_s: float = 1.0,
                 lambda_t: float = 1.0,
                 ema_alpha: float = 0.9,
                 N_prox: int = 5,
                 reduction: str = "l1"):
        super().__init__()
        self.calibration_paradigm = calibration_paradigm
        self.num_bins = num_bins
        self.lambda_s = lambda_s
        self.lambda_t = lambda_t
        self.ema_alpha = ema_alpha
        self.N_prox = N_prox
        self.reduction = reduction

        if self.calibration_paradigm == "TopLabel":
            self.register_buffer('u_s_cache', torch.zeros(num_bins))
            self.register_buffer('u_t_cache', torch.zeros(num_bins))
        elif self.calibration_paradigm == "Classwise":
            self.register_buffer('u_s_cache', torch.zeros(num_classes, num_bins))
            self.register_buffer('u_t_cache', torch.zeros(num_classes, num_bins))
        elif self.calibration_paradigm == "Canonical":
            # Determine B and anchors for simplex
            anchors_list = get_simplex_anchors(num_bins, num_classes)
            self.num_bins = len(anchors_list)  # Update to actual number of bins
            anchors = torch.tensor(anchors_list, dtype=torch.float32)
            self.register_buffer('simplex_anchors', anchors)
            
            # Cache stores vectors of size K for each bin
            self.register_buffer('u_s_cache', torch.zeros(self.num_bins, num_classes))
            self.register_buffer('u_t_cache', torch.zeros(self.num_bins, num_classes))

    def forward(self,
                train_x: torch.Tensor,
                test_x: torch.Tensor,
                train_logits: torch.Tensor,
                test_logits: torch.Tensor,
                model: nn.Module):
        
        # 1) Top-label confidences and P(Y=Ŷ|X) from classifier2
        train_probs, _ = Check_data(train_logits)
        test_probs, _ = Check_data(test_logits)

        if self.calibration_paradigm == "TopLabel":
            top_conf_train = train_probs.max(dim=1).values
            top_conf_test = test_probs.max(dim=1).values

            def head_probs_correct(h2_logits: torch.Tensor) -> torch.Tensor:
                if h2_logits.shape[1] == 1: return torch.sigmoid(h2_logits.squeeze(1))
                elif h2_logits.shape[1] == 2: return torch.softmax(h2_logits, dim=1)[:, 1]
                raise ValueError("classifier2 output shape must be (N,1) or (N,2).")

            h2_train = model.forward_classifier2(train_x) if hasattr(model, "forward_classifier2") else model.classifier2(train_x)
            h2_test = model.forward_classifier2(test_x) if hasattr(model, "forward_classifier2") else model.classifier2(test_x)
            p_correct_source = head_probs_correct(h2_train)
            p_correct_target = head_probs_correct(h2_test)

            # 2) Soft bin assignments (anchor-based)
            B = self.num_bins
            device, dtype = top_conf_test.device, top_conf_test.dtype
            anchors = (torch.arange(B, device=device, dtype=dtype) * 2 + 1) / (2.0 * B)
            soft_temp = -1.0 / (math.log(0.9) * B * B) # Heuristic from paper

            diffs_t = top_conf_test.unsqueeze(1) - anchors.unsqueeze(0)
            scores_t = - (diffs_t ** 2) / soft_temp
            w_t = torch.softmax(scores_t, dim=1) # [N_t, B]

            diffs_s = top_conf_train.unsqueeze(1) - anchors.unsqueeze(0)
            scores_s = - (diffs_s ** 2) / soft_temp
            w_s = torch.softmax(scores_s, dim=1) # [N_s, B]

            # 3) Mini-batch stats for proximal updates
            n_s_batch = w_s.sum(dim=0) # [B]
            n_t_batch = w_t.sum(dim=0) # [B]
            m_s_batch = (w_s * p_correct_source.unsqueeze(1)).sum(dim=0) # [B]
            m_t_batch = (w_t * p_correct_target.unsqueeze(1)).sum(dim=0) # [B]

            # 4) Alternating proximal updates (inner loop of Alg. 1)
            loss = torch.tensor(0.0, device=device, dtype=dtype)
            tiny = torch.finfo(dtype).eps

            u_s_new = self.u_s_cache.clone()
            u_t_new = self.u_t_cache.clone()

            def shrink(v, tau): # Proximal operator for L1 norm
                return v.sign() * torch.clamp(v.abs() - tau, min=0.0)

            # Per-bin updates
            for j in range(B):
                u_s_j, u_t_j = self.u_s_cache[j], self.u_t_cache[j]
                n_s_j, n_t_j = n_s_batch[j], n_t_batch[j]
                m_s_j, m_t_j = m_s_batch[j], m_t_batch[j]

                if n_s_j < tiny or n_t_j < tiny: continue

                w_j = n_t_j / (n_t_batch.sum() + tiny)

                # Inner proximal loop
                for _ in range(self.N_prox):
                    if self.reduction == 'l1':
                        tau_s = w_j / (2 * self.lambda_s * n_s_j)
                        v_s = m_s_j / n_s_j - u_t_j
                        u_s_j = u_t_j + shrink(v_s, tau_s)

                        tau_t = w_j / (2 * self.lambda_t * n_t_j)
                        v_t = m_t_j / n_t_j - u_s_j
                        u_t_j = u_s_j + shrink(v_t, tau_t)
                    else: # L2
                        # Corrected update rule where coefficients are consistent
                        u_s_j = (self.lambda_s*m_s_j + w_j*u_t_j) / (self.lambda_s*n_s_j + w_j)
                        u_t_j = (self.lambda_t*m_t_j + w_j*u_s_j) / (self.lambda_t*n_t_j + w_j)

                u_s_j_detached = u_s_j.detach()
                u_t_j_detached = u_t_j.detach()

                # Update cache via EMA (in-place update on the new tensor)
                u_s_new[j] = (1 - self.ema_alpha) * self.u_s_cache[j] + self.ema_alpha * u_s_j_detached
                u_t_new[j] = (1 - self.ema_alpha) * self.u_t_cache[j] + self.ema_alpha * u_t_j_detached

                # Accumulate loss terms for backpropagation (matches line 17 of Alg. 1)
                loss_s_j = (w_s[:, j] * (u_s_j_detached - p_correct_source).pow(2)).sum()
                loss_t_j = (w_t[:, j] * (u_t_j_detached - p_correct_target).pow(2)).sum()
                loss += self.lambda_s * loss_s_j + self.lambda_t * loss_t_j
            
            # Update the internal caches for the next iteration
            self.u_s_cache.copy_(u_s_new)
            self.u_t_cache.copy_(u_t_new)

            return loss
        elif self.calibration_paradigm == "Classwise":
            # 1) P(Y|X) from classifier2
            # Expect classifier2 to output K logits for K classes
            h2_train = model.forward_classifier2(train_x) if hasattr(model, "forward_classifier2") else model.classifier2(train_x)
            h2_test = model.forward_classifier2(test_x) if hasattr(model, "forward_classifier2") else model.classifier2(test_x)
            
            # Check dimensions
            if h2_train.dim() != 2:
                raise ValueError("classifier2 output must be 2D (N, K).")
            
            p_correct_source = torch.softmax(h2_train, dim=1) # [N_s, K]
            p_correct_target = torch.softmax(h2_test, dim=1)  # [N_t, K]
            
            num_classes = p_correct_source.shape[1]
            
            B = self.num_bins
            device, dtype = test_probs.device, test_probs.dtype
            anchors = (torch.arange(B, device=device, dtype=dtype) * 2 + 1) / (2.0 * B)
            soft_temp = -1.0 / (math.log(0.9) * B * B)
            tiny = torch.finfo(dtype).eps
            
            loss = torch.tensor(0.0, device=device, dtype=dtype)
            
            u_s_new = self.u_s_cache.clone()
            u_t_new = self.u_t_cache.clone()
            
            def shrink(v, tau):
                return v.sign() * torch.clamp(v.abs() - tau, min=0.0)

            # Iterate over each class
            for k in range(num_classes):
                # Confidence for class k
                conf_train_k = train_probs[:, k] # [N_s]
                conf_test_k = test_probs[:, k]   # [N_t]
                
                # True prob for class k (from classifier2)
                p_true_s_k = p_correct_source[:, k] # [N_s]
                p_true_t_k = p_correct_target[:, k] # [N_t]
                
                # Soft binning based on conf_test_k
                diffs_t = conf_test_k.unsqueeze(1) - anchors.unsqueeze(0)
                scores_t = - (diffs_t ** 2) / soft_temp
                w_t = torch.softmax(scores_t, dim=1) # [N_t, B]

                diffs_s = conf_train_k.unsqueeze(1) - anchors.unsqueeze(0)
                scores_s = - (diffs_s ** 2) / soft_temp
                w_s = torch.softmax(scores_s, dim=1) # [N_s, B]
                
                # Batch stats
                n_s_batch = w_s.sum(dim=0)
                n_t_batch = w_t.sum(dim=0)
                m_s_batch = (w_s * p_true_s_k.unsqueeze(1)).sum(dim=0)
                m_t_batch = (w_t * p_true_t_k.unsqueeze(1)).sum(dim=0)
                
                # Proximal updates for this class
                for j in range(B):
                    u_s_j, u_t_j = self.u_s_cache[k, j], self.u_t_cache[k, j]
                    n_s_j, n_t_j = n_s_batch[j], n_t_batch[j]
                    m_s_j, m_t_j = m_s_batch[j], m_t_batch[j]
                    
                    if n_s_j < tiny or n_t_j < tiny: continue
                    
                    w_j = n_t_j / (n_t_batch.sum() + tiny)
                    
                    for _ in range(self.N_prox):
                        if self.reduction == 'l1':
                            tau_s = w_j / (2 * self.lambda_s * n_s_j)
                            v_s = m_s_j / n_s_j - u_t_j
                            u_s_j = u_t_j + shrink(v_s, tau_s)

                            tau_t = w_j / (2 * self.lambda_t * n_t_j)
                            v_t = m_t_j / n_t_j - u_s_j
                            u_t_j = u_s_j + shrink(v_t, tau_t)
                        else: # L2
                            u_s_j = (self.lambda_s*m_s_j + w_j*u_t_j) / (self.lambda_s*n_s_j + w_j)
                            u_t_j = (self.lambda_t*m_t_j + w_j*u_s_j) / (self.lambda_t*n_t_j + w_j)
                    
                    u_s_j_detached = u_s_j.detach()
                    u_t_j_detached = u_t_j.detach()
                    
                    # Update cache
                    u_s_new[k, j] = (1 - self.ema_alpha) * self.u_s_cache[k, j] + self.ema_alpha * u_s_j_detached
                    u_t_new[k, j] = (1 - self.ema_alpha) * self.u_t_cache[k, j] + self.ema_alpha * u_t_j_detached
                    
                    # Loss
                    loss_s_j = (w_s[:, j] * (u_s_j_detached - p_true_s_k).pow(2)).sum()
                    loss_t_j = (w_t[:, j] * (u_t_j_detached - p_true_t_k).pow(2)).sum()
                    loss += self.lambda_s * loss_s_j + self.lambda_t * loss_t_j

            self.u_s_cache.copy_(u_s_new)
            self.u_t_cache.copy_(u_t_new)
            
            return loss / num_classes
        elif self.calibration_paradigm == "Canonical":
            # 1) P(Y|X) from classifier2
            h2_train = model.forward_classifier2(train_x) if hasattr(model, "forward_classifier2") else model.classifier2(train_x)
            h2_test = model.forward_classifier2(test_x) if hasattr(model, "forward_classifier2") else model.classifier2(test_x)
            
            p_correct_source = torch.softmax(h2_train, dim=1) # [N_s, K]
            p_correct_target = torch.softmax(h2_test, dim=1)  # [N_t, K]
            num_classes = p_correct_source.shape[1]

            # 2) Binning based on vector distance to anchors
            B = self.num_bins
            device, dtype = train_probs.device, train_probs.dtype
            anchors = self.simplex_anchors.to(device)
            
            soft_temp = -1.0 / (math.log(0.9) * B * B)
            tiny = torch.finfo(dtype).eps
            
            # (N, B)
            diffs_t = test_probs.unsqueeze(1) - anchors.unsqueeze(0)
            scores_t = - diffs_t.pow(2).sum(dim=2) / soft_temp
            w_t = torch.softmax(scores_t, dim=1) 

            diffs_s = train_probs.unsqueeze(1) - anchors.unsqueeze(0)
            scores_s = - diffs_s.pow(2).sum(dim=2) / soft_temp
            w_s = torch.softmax(scores_s, dim=1) 
            
            # 3) Batch stats (Vectorized)
            n_s_batch = w_s.sum(dim=0) # [B]
            n_t_batch = w_t.sum(dim=0) # [B]
            
            m_s_batch = (w_s.unsqueeze(2) * p_correct_source.unsqueeze(1)).sum(dim=0) # [B, K]
            m_t_batch = (w_t.unsqueeze(2) * p_correct_target.unsqueeze(1)).sum(dim=0) # [B, K]
            
            loss = torch.tensor(0.0, device=device, dtype=dtype)
            
            u_s_new = self.u_s_cache.clone()
            u_t_new = self.u_t_cache.clone()
            
            def shrink(v, tau):
                return v.sign() * torch.clamp(v.abs() - tau, min=0.0)

            for j in range(B):
                u_s_j, u_t_j = self.u_s_cache[j], self.u_t_cache[j] # [K]
                n_s_j, n_t_j = n_s_batch[j], n_t_batch[j] # scalar
                m_s_j, m_t_j = m_s_batch[j], m_t_batch[j] # [K]
                
                if n_s_j < tiny or n_t_j < tiny: continue
                
                w_j = n_t_j / (n_t_batch.sum() + tiny)
                
                for _ in range(self.N_prox):
                    if self.reduction == 'l1':
                        tau_s = w_j / (2 * self.lambda_s * n_s_j)
                        v_s = m_s_j / n_s_j - u_t_j
                        u_s_j = u_t_j + shrink(v_s, tau_s) 

                        tau_t = w_j / (2 * self.lambda_t * n_t_j)
                        v_t = m_t_j / n_t_j - u_s_j
                        u_t_j = u_s_j + shrink(v_t, tau_t)
                    else: # L2
                        u_s_j = (self.lambda_s*m_s_j + w_j*u_t_j) / (self.lambda_s*n_s_j + w_j)
                        u_t_j = (self.lambda_t*m_t_j + w_j*u_s_j) / (self.lambda_t*n_t_j + w_j)
                
                u_s_j_detached = u_s_j.detach()
                u_t_j_detached = u_t_j.detach()
                
                u_s_new[j] = (1 - self.ema_alpha) * self.u_s_cache[j] + self.ema_alpha * u_s_j_detached
                u_t_new[j] = (1 - self.ema_alpha) * self.u_t_cache[j] + self.ema_alpha * u_t_j_detached
                
                loss_s_j = (w_s[:, j] * (u_s_j_detached.unsqueeze(0) - p_correct_source).pow(2).sum(dim=1)).sum()
                loss_t_j = (w_t[:, j] * (u_t_j_detached.unsqueeze(0) - p_correct_target).pow(2).sum(dim=1)).sum()
                loss += self.lambda_s * loss_s_j + self.lambda_t * loss_t_j

            self.u_s_cache.copy_(u_s_new)
            self.u_t_cache.copy_(u_t_new)
            
            return loss/num_classes

