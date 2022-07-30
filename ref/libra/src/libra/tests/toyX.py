
x01 = float(input())
x02 = float(input())

x11 = -0.31*x01 + 0.99*x02 -0.63
x12 = -1.25*x01 - 0.64*x02 + 1.88
#
x11 = x11 if x11 >= 0 else 0
x12 = x12 if x12 >= 0 else 0

x21 = 0.40*x11 + 1.21*x12 + 0.00
x22 = 0.64*x11 + 0.69*x12 - 0.39
#
x21 = x21 if x21 >= 0 else 0
x22 = x22 if x22 >= 0 else 0

x31 = 0.26*x21 + 0.33*x22 + 0.45
x32 = 1.42*x21 + 0.40*x22 - 0.45
#
if x31 < x32:
    print('x32')
elif x32 < x31:
    print('x31')
else:
    raise ValueError
