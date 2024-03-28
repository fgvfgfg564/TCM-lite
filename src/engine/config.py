from typing_extensions import TypeAlias, Literal, Union, Type, Optional, TypedDict

from dataclasses import dataclass

OptionalFloat: TypeAlias = Optional[float]


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


ConstraintConfig: TypeAlias = OptimalFloatConfig
BalancerConfig: TypeAlias = OptimalFloatConfig

MULTI_OBJECT = BalancerConfig()


class MOConfig:
    """
    Target = R + \lambda * D + \alpha * T
    """

    def __init__(
        self,
        object_config: ObjectConfig = ObjectConfig(),
        constraints: ConstraintConfig = ConstraintConfig(),
        balancers: BalancerConfig = BalancerConfig(),
    ) -> None:
        """
        balancers: manually assign the balance value between values
        limiters: limit the maximum value of certain variable
        """
        self.balancers = balancers
        self.constraints = constraints
        self.object_config = object_config
        self.multiobject = balancers == MULTI_OBJECT
        self._self_check()

    def _check_value(
        self,
        is_object: bool,
        balance_value: OptionalFloat,
        name,
    ) -> int:
        if not is_object and balance_value is not None:
            raise ValueError(f"Non-object '{name}' does not require a balancer.")
        if balance_value is not None and self.multiobject:
            raise ValueError(
                f"Multi object mode requires no balancer for object '{name}'"
            )
        if is_object and balance_value is None and not self.multiobject:
            raise ValueError(
                f"Single object mode requires object '{name}' have a balancer"
            )
        if is_object:
            return 1
        else:
            return 0

    def _self_check(self):
        # R should be limited or have balancer
        variable_count = 0
        variable_count += self._check_value(self.object_config.R, self.balancers.R, "R")
        variable_count += self._check_value(self.object_config.D, self.balancers.D, "D")
        variable_count += self._check_value(self.object_config.T, self.balancers.T, "T")

        if variable_count == 0:
            raise ValueError("Requires at least 1 variable.")

    def __repr__(self) -> str:
        ifmo = "Multi Object" if self.multiobject else "Single Object"
        objectives = []
        if self.multiobject:
            if self.object_config.R:
                objectives.append("R")
            if self.object_config.D is not None:
                objectives.append("D")
            if self.object_config.T is not None:
                objectives.append("T")
        else:
            if self.object_config.R:
                objectives.append(f"{self.balancers.R} * R")
            if self.object_config.D:
                objectives.append(f"{self.balancers.D} * D")
            if self.object_config.T:
                objectives.append(f"{self.balancers.T} * T")
            objectives = " + ".join(objectives)

        constraints = []
        if self.constraints.R is not None:
            constraints.append(f"R <= {self.constraints.R}")
        if self.constraints.D is not None:
            constraints.append(f"D <= {self.constraints.D}")
        if self.constraints.T is not None:
            constraints.append(f"T <= {self.constraints.T}")

        return f"""{ifmo}
Objectives: {objectives}
Constraints: {constraints}
"""
