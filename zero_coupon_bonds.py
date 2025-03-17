
class ZeroCouponBonds:
    def __init__(self, principal, maturity, interest_rate):
        self.principal = principal
        self.maturity = maturity
        self.interest_rate = interest_rate/100

    def present_value(self, x, n):
        return x/(1 + self.interest_rate)**n

    def calculate_price(self):
        return self.present_value(self.principal, self.maturity)


if __name__ == '__main__':
    bond = ZeroCouponBonds(1000,2,4)
    print("Price of the bond in dollars: %.2f" % bond.calculate_price())
    print(bond.maturity)

