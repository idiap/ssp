set xrange [0:1]
set yrange [0:1]
Tp=0.40
Tn=0.16

# Polynomial
p1(t) = 3.0*(t/Tp)**2 - 2.0*(t/Tp)**3
p2(t) = 1.0 - ((t-Tp)/Tn)**2

# Trig
t1(t) = 0.5*(1.0-cos(pi*t/Tp))
t2(t) = cos((t-Tp)/Tn * pi/2)

# Gamma
ga = 4
gB = 0.1/(ga-1)
g(t) = exp((ga-1)*log(1.0-t) - (1.0-t)/gB)

# Inv Gamma
ia = 4
iB = 0.1*(ia+1)
ig(t) = exp(-(ia+1)*log(1.0-t) - iB/(1.0-t))

plot p1(x), p2(x), t1(x), t2(x), \
     g(x) / g(1.0-gB*(ga-1)), \
     ig(x) / ig(1.0-iB/(ia+1))

pause -1