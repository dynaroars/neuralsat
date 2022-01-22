from z3 import *
import z3
import z3.z3core as z3core




def test():
    y = z3.Real('y')
    z = z3.Real('z')
    return y - 3*z

def main():
    b = z3.Real('k')
    c = z3.Real('c')
    t = test(
        )
    print(z3.substitute(t, (b, 2*c)))


if __name__ == '__main__':
    main()