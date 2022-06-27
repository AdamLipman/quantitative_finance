from math import exp

def future_discrete_value(x,r,n):
    return x*(1+r)**n

def present_discrete_value(x,r,n):
    return x*(1+r)**(-n)

def future_continuous_value(x,r,t):
    return x*exp(r*t)

def present_continuous_value(x,r,t):
    return x*exp(-r*t)

if __name__ == '__main__':

    #value of investment in dollars
    x=1000
    #define the interest rate
    r=0.1
    #duration (years)
    n=3

    print('Future values of x: %s' % future_discrete_value(x,r,n))
    print('Present values of x: %s' % present_discrete_value(x, r, n))
    print('Future values of x: %s' % future_continuous_value(x, r, n))
    print('Present values of x: %s' % present_continuous_value(x, r, n))