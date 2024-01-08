from beartype import beartype

INPUT_SPLIT_RESTART_STRATEGIES = [
    # {'abstract_method': 'crown-optimized', 'split_method': 'naive'},
    {'abstract_method': 'forward+backward', 'split_method': 'naive'},
    {'abstract_method': 'backward',        'split_method': 'naive'},
    {'abstract_method': 'crown-optimized', 'split_method': 'naive'},
    # {'abstract_method': 'crown-optimized', 'split_method': 'gradient'},
]

HIDDEN_SPLIT_RESTART_STRATEGIES = [
    {'abstract_method': 'crown-optimized', 'decision_topk': 10, 'random_selection': False},
    {'abstract_method': 'crown-optimized', 'decision_topk': 20, 'random_selection': False},
    {'abstract_method': 'crown-optimized', 'decision_topk': 30, 'random_selection': False},
    # {'abstract_method': 'crown-optimized', 'decision_topk': 30, 'random_selection': True},
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
    
    
        
    