import backtrader as bt

class CustomPandasData(bt.feeds.PandasData):
    lines = ('close', 'custom')
    params = (
        ('datetime', None),
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', None),
        ('custom', 'prediction')
    )

class CustomIndicator(bt.Indicator):
    lines = ('custom',)

    def __init__(self):
        self.lines.custom = self.data.custom


class CustomStrategy(bt.Strategy):
    def __init__(self):
        self.stock_price = self.data.close
        self.custom_indicator = CustomIndicator(subplot=True)

    def notify_order(self, order):
        # if order.status == order.Completed:
        #     if order.isbuy():
        #         print(f'[{self.data.datetime.date(0)}] {"Buy":>4} @ {order.executed.price:.2f},\tCost: {order.executed.value:.2f},\tCommision: {order.executed.comm:.2f}')
        #     elif order.issell():
        #         print(f'[{self.data.datetime.date(0)}] {"Sell":>3} @ {order.executed.price:.2f},\tCost: {order.executed.value:.2f},\tCommision: {order.executed.comm:.2f}')
        # elif order.status in [order.Canceled, order.Margin, order.Rejected]:
        #     print('Order Canceled/Margin/Rejected')
        self.order = None

    def notify_trade(self, trade):
        pass
        # if trade.isclosed:
        #     print(f'Closed a position, Operational Profit, Gross: {trade.pnl:.2f}, Net Profit: {trade.pnlcomm:.2f}')

    def next(self):
        if self.custom_indicator == 1:
            self.order = self.buy()
        elif self.position and self.custom_indicator == 0: # Disabled short selling
            self.order = self.sell()