from typing import Dict, Set, Iterable

def compute_entity_f05(gt_set: Set[str], pred_set: Set[str]) -> float:
    """Compute F_0.5 score for a single Source 1 entity.
    
    Handles singletons (empty ground truth) and general matches.
    Precision is weighted 2x over Recall:
        F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    """
    if len(gt_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    
    if len(pred_set) == 0:
        return 0.0
    
    tp = len(gt_set & pred_set)
    if tp == 0:
        return 0.0
    
    prec = tp / len(pred_set)
    rec = tp / len(gt_set)
    
    denom = 0.25 * prec + rec
    if denom == 0.0:
        return 0.0
    return (1.25 * prec * rec) / denom

def compute_macro_f05(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]]) -> float:
    """Compute the macro-averaged F_0.5 score across all Source 1 entities."""
    total_score = 0.0
    n = len(ground_truth)
    if n == 0:
        return 0.0
    
    for s1_id, gt_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        total_score += compute_entity_f05(gt_set, pred_set)
        
    return total_score / n
