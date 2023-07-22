import copy

class Tightener:
    
    def __init__(self, abstractor):
        self.abstractor = abstractor
        
        
    def __call__(self, domain_params):
        return domain_params