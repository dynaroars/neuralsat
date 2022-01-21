from z3 import *
import z3
import z3.z3core as z3core




def main():
    y = z3.Real('y')
    z = z3.Real('z')

    t = z3.simplify(z3.substitute(-y + 1, (y, z-2))) <= 0
    print(dir(t))
    print(vars(t))
    print(vars(t.ctx))
    print(dir(t.ctx.eh))

    print(str(t).replace('*', ''))


if __name__ == '__main__':
    main()