from beartype import beartype

INPUT_SPLIT_RESTART_STRATEGIES = [
    {'input_split': True, 'abstract_method': 'forward+backward', 'decision_method': 'smart', 'decision_topk': 1},
    {'input_split': True, 'abstract_method': 'forward+backward', 'decision_method': 'naive', 'decision_topk': 1},
    
    {'input_split': True, 'abstract_method': 'backward',         'decision_method': 'smart', 'decision_topk': 1},
    {'input_split': True, 'abstract_method': 'backward',         'decision_method': 'naive', 'decision_topk': 1},
    
    {'input_split': True, 'abstract_method': 'crown-optimized',  'decision_method': 'smart', 'decision_topk': 1},
    {'input_split': True, 'abstract_method': 'crown-optimized',  'decision_method': 'naive', 'decision_topk': 1},
]

HIDDEN_SPLIT_RESTART_STRATEGIES = [
    {'input_split': False, 'abstract_method': 'crown-optimized', 'decision_method': 'smart', 'decision_topk': 10},
    {'input_split': False, 'abstract_method': 'crown-optimized', 'decision_method': 'smart', 'decision_topk': 20},
    {'input_split': False, 'abstract_method': 'crown-optimized', 'decision_method': 'smart', 'decision_topk': 30},
]


@beartype
def get_restart_strategy(nth_restart: int, input_split: bool = False) -> dict:
    if input_split:
        if nth_restart >= len(INPUT_SPLIT_RESTART_STRATEGIES):
            strategy = INPUT_SPLIT_RESTART_STRATEGIES[-1]
        else:
            strategy = INPUT_SPLIT_RESTART_STRATEGIES[nth_restart]
    else:
        if nth_restart >= len(HIDDEN_SPLIT_RESTART_STRATEGIES):
            strategy = HIDDEN_SPLIT_RESTART_STRATEGIES[-1]
        else:
            strategy = HIDDEN_SPLIT_RESTART_STRATEGIES[nth_restart]
    return strategy
    
    
        
    