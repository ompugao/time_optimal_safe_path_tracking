"""1-D reconstruction of the SSM controller of Zanchettin et al., IEEE T-ASE 13(2), 2016.

FIXME: the full text was not accessible, so this follows how arXiv 2306.05197
describes it: each cycle an LP picks the next velocity (scaling of the nominal
motion) under hard acceleration limits and the stopping-time constraint (9b),
  T_s * (v_robot + v_human) <= d - d_min,
which bounds the velocity from above. When braking at full deceleration cannot
meet (9b), the LP is infeasible; we then brake at full deceleration and count it
(steps where the car is already stopped are not counted).
"""

import math
from dataclasses import dataclass


@dataclass
class ZanchettinCar:
    vmax: float
    amax: float
    goal: float
    stopping_time: float
    human_speed: float
    min_distance: float = 0.0
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    infeasible_steps: int = 0

    def safety_bound(self, distance):
        return (distance - self.min_distance) / self.stopping_time - self.human_speed

    def goal_bound(self, dt):
        """Not part of the baseline: lets the car stop at the end of the path."""
        remaining = max(self.goal - self.position - self.velocity * dt, 0.0)
        return math.sqrt(2.0 * self.amax * remaining)

    def step(self, distance, dt):
        lowest = max(self.velocity - self.amax * dt, 0.0)
        safety = self.safety_bound(distance)
        if safety < lowest and lowest > 0.0:
            self.infeasible_steps += 1
        upper = min(self.vmax, self.velocity + self.amax * dt, safety, self.goal_bound(dt))
        new_velocity = max(upper, lowest)
        self.acceleration = (new_velocity - self.velocity) / dt
        self.position = min(self.position + 0.5 * (self.velocity + new_velocity) * dt, self.goal)
        self.velocity = new_velocity
