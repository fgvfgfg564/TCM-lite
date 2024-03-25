from typing_extensions import TypeAlias, Literal, Union, Type

from dataclasses import dataclass

OptionalFloat: TypeAlias = float | None


@dataclass
class OptimalFloatConfig:
    R: OptionalFloat = None
    D: OptionalFloat = None
    T: OptionalFloat = None


@dataclass
class ObjectConfig:
    R: bool = True
    D: bool = True
    T: bool = True


LimiterConfig: TypeAlias = OptimalFloatConfig
BalancerConfig: TypeAlias = OptimalFloatConfig

MULTI_OBJECT = BalancerConfig()


class MOConfig:
    """
    Target = R + \lambda * D + \alpha * T
    """

    def __init__(
        self,
        object_config: ObjectConfig = ObjectConfig(),
        limiters: LimiterConfig = LimiterConfig(),
        balancers: BalancerConfig = BalancerConfig(),
    ) -> None:
        """
        balancers: manually assign the balance value between values
        limiters: limit the maximum value of certain variable
        """
        self.balancers = balancers
        self.limiters = limiters
        self.object_config = object_config
        self.multiobject = balancers == MULTI_OBJECT

    def _check_value(
        self,
        is_object: bool,
        limit_value: OptionalFloat,
        balance_value: OptionalFloat,
        name,
    ) -> int:
        if not is_object and balance_value is not None:
            raise ValueError(f"Non-object '{name}' does not require a balancer.")
        if balance_value is not None and self.multiobject:
            raise ValueError(
                f"Multi object mode requires no balancer for object '{name}'"
            )
        if limit_value is None and balance_value is None and not self.multiobject:
            raise ValueError(
                f"Single object mode requires object '{name}' have a limit or balancer"
            )
        # if limit_value is None:
        #     return 0
        # else:
        #     return 1

    def _self_check(self):
        # R should be limited or have balancer
        variable_count = 0
        variable_count += self._check_value(
            self.object_config.R, self.limiters.R, self.balancers.R, "R"
        )
        variable_count += self._check_value(
            self.object_config.R, self.limiters.D, self.balancers.D, "D"
        )
        variable_count += self._check_value(
            self.object_config.R, self.limiters.T, self.balancers.T, "T"
        )

        if variable_count == 0:
            raise ValueError("Requires at least 1 variable.")
