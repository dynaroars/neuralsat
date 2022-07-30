
x11 = x01 + 2*x02 - 1 # x01 + x02 (with x02 := 2*x02 -1)
x12 = x01 - 2*x02 + 1 # x01 - x02 (with x02 := 2*x02 -1)

ReLU(x11)
ReLU(x12)

x21 = x11 + x12 + 0
x22 = x11 - x12 + 0

ReLU(x21)
ReLU(x22)

x31 = x21 + x22 + 1
x32 = 0*x21 + x22 + 0
