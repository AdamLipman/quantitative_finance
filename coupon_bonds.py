

class CouponBond:
    def __init__(self, principal, rate, maturity, interest_rate):
        self.principal = principal
        self.rate = rate/100
        self.maturity = maturity
        self.interest_rate = interest_rate/100

    
    def present_value(self, x, n):
        return x/(1+self.interest_rate)**n


    def calculate_price(self):
        price = 0
        #discount the coupon payments
        for t in range(1, self.maturity+1):
            price += self.present_value(self.principal*self.rate, t)

        #discount principal amount
        price += self.present_value(self.principal, self.maturity)

        return price

if __name__=='__main__':
    bond = CouponBond(1000, 10, 3, 4)
    print('Bond price: %.2f' % bond.calculate_price())
