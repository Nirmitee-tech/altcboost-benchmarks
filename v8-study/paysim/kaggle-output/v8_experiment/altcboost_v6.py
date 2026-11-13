"""Direct pairwise temporal-tail candidate; inherits the checked-descent trainer."""
from altcboost_v5 import ALTCBoostV5
from tail_objective import pairwise_tail_objective

class ALTCBoostV6(ALTCBoostV5):
    def objective(self,y,score,groups):
        return pairwise_tail_objective(y,score,groups,self.tail_fraction,self.temperature,
                                      self.group_temperature,self.anchor_weight)
