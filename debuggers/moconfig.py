from src.engine.config import *

object_config = ObjectConfig(R=True, D=True, T=True)
constraints = ConstraintConfig(R=13850, D=None, T=15)
balancers = BalancerConfig()
config = MOConfig(object_config, constraints, balancers)
print(config)

balancers = BalancerConfig(D=1.8)
config = MOConfig(object_config, constraints, balancers)
print(config)
