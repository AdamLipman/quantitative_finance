from math import exp

class Discount_Coupon_Bond:
    def __init__(self, principal, coupon_rate, market_rate, maturity):
        self.principal = principal
        self.coupon_rate = coupon_rate/100
        self.market_rate = market_rate/100
        self.maturity = maturity

    def present_value(self, A, t):
        return A*exp(-self.market_rate*t)

    def calculate_discount_value(self):
        price = 0
        for i in range(1, self.maturity+1):
            price += self.present_value(self.principal*self.coupon_rate, i)

        price += self.present_value(self.principal, self.maturity)
        return price

if __name__ == '__main__':
    bond = Discount_Coupon_Bond(1000, 10, 4, 3)
    print('Bond discount price = %.2f' % bond.calculate_discount_value())

