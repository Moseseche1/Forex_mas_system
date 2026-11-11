"""Deep reinforcement learning is not enabled in this safe baseline."""
class DRLAdaptor:
    def __init__(self,*args,**kwargs):
        raise RuntimeError("DRL model training/inference is disabled until a validated dataset and evaluation protocol are supplied")
class DeepAdaptiveLearner:
    def __init__(self,*args,**kwargs):
        self.enabled=False
    def remember(self,*args,**kwargs):
        raise RuntimeError("DRL learning is disabled")
    def act(self,*args,**kwargs):
        raise RuntimeError("DRL actions are disabled")
    def replay(self,*args,**kwargs):
        raise RuntimeError("DRL training is disabled")
