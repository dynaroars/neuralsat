from core.abstraction.third_party.abcrown.abcrown_abstraction import ABCrownAbstraction

class Abstractor:

    def __init__(self, net, spec):
        self.net = net
        self.spec = spec
        self.core = ABCrownAbstraction(net, spec)

    def forward(self, input_lower, input_upper, extra_params=None):
        domains = self.core.forward(input_lower=input_lower, input_upper=input_upper, extra_params=extra_params)
        return domains