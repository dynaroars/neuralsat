

class GlobalSettings:

    def __init__(self):
        
        # restart thresholds, if exceed, restart
        self.max_hidden_branches = 1e5 
        self.max_hidden_visited_branches = 1e5 
        
        self.max_input_branches  = 1e5
        self.max_input_visited_branches = 5e6
        
        # attack
        self.use_attack = 1
        
        # restart
        self.use_restart = 1
        
        # theshold for switching between input and hidden splitting
        self.safety_property_threshold = 0.15

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


Settings = GlobalSettings()