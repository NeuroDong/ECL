import numpy as np
import math

def equal_interval_ece(probs, labels, num_bins=15):
    '''
    equal interval binning
    '''
    bins = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    
    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        bin_count = np.sum(in_bin)
        if bin_count > 5:
            avg_confidence = np.mean(probs[in_bin])
            avg_accuracy = np.mean(labels[in_bin])
            bin_accs.append(avg_accuracy)
            bin_confs.append(avg_confidence)
            bin_counts.append(bin_count)
            ece += (bin_count / len(probs)) * np.abs(avg_accuracy - avg_confidence)
        else:
            bin_accs.append(0)
            bin_confs.append(0)
            bin_counts.append(0)
    return ece, bins, bin_confs, bin_accs, bin_counts

def classwise_ece(probs, labels, num_bins=15, plot_class_idx=1):
    '''
    Calculate Classwise ECE and return plotting data for a specific class.
    probs: (N, K) numpy array
    labels: (N,) numpy array of true class indices
    '''
    num_classes = probs.shape[1]
    ece_sum = 0.0
    
    plot_data = None
    
    for k in range(num_classes):
        # Binary problem for class k
        # Confidence that label is k
        probs_k = probs[:, k]
        # True label is k?
        labels_k = (labels == k).astype(int)
        
        # Reuse equal_interval_ece logic
        bins = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]
        
        ece_k = 0.0
        bin_accs = []
        bin_confs = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probs_k > bin_lower) & (probs_k <= bin_upper)
            bin_count = np.sum(in_bin)
            if bin_count > 5:
                avg_confidence = np.mean(probs_k[in_bin])
                avg_accuracy = np.mean(labels_k[in_bin])
                bin_accs.append(avg_accuracy)
                bin_confs.append(avg_confidence)
                bin_counts.append(bin_count)
                ece_k += (bin_count / len(probs_k)) * np.abs(avg_accuracy - avg_confidence)
            else:
                bin_accs.append(0)
                bin_confs.append(0)
                bin_counts.append(0)
        
        ece_sum += ece_k
        
        if k == plot_class_idx:
            plot_data = (bins, bin_confs, bin_accs, bin_counts)
            
    avg_ece = ece_sum / num_classes
    return avg_ece, plot_data[0], plot_data[1], plot_data[2], plot_data[3]

def canonical_ece(probs, labels, num_bins=15):
    """
    Calculate Canonical ECE using Simplex Anchors (Centroids) L2 distance binning.
    Returns:
        ece: Scalar ECE value (weighted average of L2 norm errors).
        ece_in_anchors: List of L2 errors for each bin (aligned with anchors).
        counts_in_anchors: List of sample counts for each bin.
    """
    N_samples = probs.shape[0]
    num_classes = probs.shape[1]
    
    # One-hot encoding labels if needed
    if labels.ndim == 1:
        labels_onehot = np.zeros((N_samples, num_classes))
        labels_onehot[np.arange(N_samples), labels] = 1
    else:
        labels_onehot = labels
        
    # --- 1. Generate Anchors (Centroids) ---
    # Find resolution N such that number of anchors >= num_bins
    N = 1
    while True:
        count = math.comb(N + num_classes - 1, num_classes - 1)
        if count >= num_bins:
            break
        N += 1
        if N > 50: break 
    
    def recursive_gen(n, dim):
        if dim == 1:
            return [[n]]
        res = []
        for i in range(n + 1):
            sub_res = recursive_gen(n - i, dim - 1)
            for sub in sub_res:
                res.append([i] + sub)
        return res

    grid = np.array(recursive_gen(N, num_classes), dtype=np.float32)
    # Use centroids formula: (grid + 1/K) / (N + 1)
    anchors = (grid + 1.0 / num_classes) / (float(N) + 1.0)
    num_anchors = len(anchors)
    
    # --- 2. Assign Samples to Nearest Anchor (L2 distance) ---
    assignments = []
    # Process in chunks to avoid huge memory usage if N_samples is large
    chunk_size = 1000 
    for i in range(0, N_samples, chunk_size):
        chunk_probs = probs[i:i+chunk_size]
        # [Chunk, 1, K] - [1, Anchors, K] -> [Chunk, Anchors, K]
        # L2 distance squared
        dists = np.sum((chunk_probs[:, np.newaxis, :] - anchors[np.newaxis, :, :])**2, axis=2)
        chunk_assigns = np.argmin(dists, axis=1)
        assignments.append(chunk_assigns)
    
    assignments = np.concatenate(assignments)
    
    # --- 3. Compute ECE stats per bin ---
    ece = 0.0
    ece_in_anchors = [] # Error per bin
    counts_in_anchors = [] # Count per bin
    
    for i in range(num_anchors):
        mask = (assignments == i)
        count = np.sum(mask)
        counts_in_anchors.append(count)
        
        if count > 0:
            avg_conf_vec = np.mean(probs[mask], axis=0) # [K]
            avg_acc_vec = np.mean(labels_onehot[mask], axis=0) # [K]
            
            # L2 norm of the difference vector
            error = np.linalg.norm(avg_conf_vec - avg_acc_vec)
            ece_in_anchors.append(error)
            
            ece += (count / N_samples) * error
        else:
            ece_in_anchors.append(0.0)
            
    return ece, ece_in_anchors, counts_in_anchors

