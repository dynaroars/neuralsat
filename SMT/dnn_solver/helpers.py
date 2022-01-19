
class Utils:

    def And(term1, term2):
        return f'(and {term1} {term2})'

    def Not(term1):
        return f'(not {term1})'

    def Or(term1, term2):
        return f'(or {term1} {term2})'

    def Prove(term1, term2):
        return Utils.And(term1, Utils.Not(term2))
