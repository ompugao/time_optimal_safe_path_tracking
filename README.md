# Time-optimal safe path tracking (reproduction)

A from-scratch reproduction of
S. Fujii and Q.-C. Pham, *Time-Optimal Path Tracking with ISO Safety Guarantees*, IROS 2023
([arXiv:2306.05197](https://arxiv.org/abs/2306.05197)).

The robot tracks a fixed path as fast as its joint limits allow, but chooses each control
so that it can always be at rest before any obstacle (moving at most `v_max`) can reach it.
If a collision happens, the robot is stationary (ISO/TS 15066 speed and separation monitoring).

## Setup

```sh
uv sync
./scripts/fetch_assets.sh      # UR5e model from MuJoCo Menagerie (only for the robot experiment)
uv run pytest
```

Torch comes from the CUDA 12.8 wheel index (needed for Blackwell GPUs). Every torch function
also runs with `device="cpu"`.

## Layout

| file | content |
|---|---|
| `src/tospt/constraints.py` | per-stage `a u + b x <= c` from toppra joint velocity/acceleration constraints |
| `src/tospt/lp2d.py` | one-step sets `Q_i` (2-variable LP) |
| `src/tospt/stoppable_sets.py` | stoppable sets `K[j, i]` |
| `src/tospt/lp1d.py` | branch-free 1-D LP (paper Sec. III-C) |
| `src/tospt/time_to_reach.py` | Time-to-Reach `tau[j, i, k]` and next index `rho` (Alg. 1); serial and torch |
| `src/tospt/controller.py` | Time-to-Arrive (Eq. 6), stop-stage selection (Eq. 7), greedy step, path follower |
| `src/tospt/mujoco_scene.py` | UR5e scene (optional payload), distances to its collision capsules |
| `src/tospt/dynamics.py` | torque limits from MuJoCo inverse dynamics, torque actuators, computed-torque tracking |
| `experiments/car_1d.py`, `zanchettin.py` | Exp. 1: 1-D car vs. Zanchettin-style controller |
| `experiments/bench_precompute.py` | Exp. 2: serial vs. GPU pre-computation |
| `experiments/robot_6dof.py` | Exp. 3: UR5e in MuJoCo with 1 or 6 obstacles at 1.6 m/s |
| `experiments/robot_6dof_dynamics.py` | demo: Exp. 3 with torque limits and a physically simulated robot |

## Running the experiments

```sh
uv run python experiments/car_1d.py                     # -> results/car_1d.png
uv run python experiments/car_1d.py --sweep-stopping-time
uv run python experiments/bench_precompute.py           # -> results/bench_precompute.png (~2 min)
uv run python experiments/robot_6dof.py --obstacles 1   # add --viewer to watch in MuJoCo
uv run python experiments/robot_6dof.py --obstacles 6
uv run python experiments/robot_6dof.py --precompute-trials 20
uv run python experiments/robot_6dof_dynamics.py --obstacles 6                      # add --viewer to watch
uv run python experiments/robot_6dof_dynamics.py --obstacles 6 --actual-payload 4.0 # model error
```

## Results

Hardware: NVIDIA RTX 5080 Laptop GPU. The paper used a GTX 1080 Ti (Exp. 2) and an RTX 2060 Max-Q (Exp. 3).

### Exp. 1: 1-D car (goal 25 m, 20 m/s, 100 m/s², wall 20 m/s, `d_protective = 0`)

| | paper | reproduced |
|---|---|---|
| ours stops at contact with max deceleration | yes | yes (speed 0 while touching) |
| ours reaches the goal first | yes | 2.82 s vs. 3.58 s (`T_s = 0.55 s`) |
| smallest feasible stopping time `T_s` of the baseline | 0.55 s | 0.40 s (= `(v_max + v_wall) / a_max`) |
| bang-bang spikes when following the wall | yes | yes |

### Exp. 2: Time-to-Reach computation (random 6-DoF paths, 10 trials)

| setting | serial | torch CUDA float64 |
|---|---|---|
| N = 300, M = 50 | 1.01 s | 0.052 s |
| N = 500, M = 50 | 2.16 s | 0.142 s |

The paper reports 0.25 s for N = 500, M = 30 on the GPU, and at least 1 s for serial.

### Exp. 3: UR5e, N = 513 stages, ±20 rad/s², ±π rad/s, 20 s runs, control period 4 ms

| | paper | reproduced |
|---|---|---|
| total pre-computation, 20 paths, serial | 4.03 ± 1.01 s | 4.93 ± 1.23 s |
| total pre-computation, 20 paths, GPU | 0.40 ± 0.09 s | 0.21 ± 0.01 s |
| stoppable sets | 0.20 ± 0.04 s (16 processes) | 0.12 s (1 process) |
| execution phase per cycle | within 6 ms | max 3.9–6.3 ms, mean 0.9–1.6 ms (with obstacles) |
| joint speed whenever distance <= 0 | 0 | 0 (all runs) |

| obstacles | M | legs done in 20 s | cycles in contact | emergency decisions |
|---|---|---|---|---|
| 0 | 30 | 31 (= exact TOPP-RA time, 0.64 s per leg) | 0 | 0 |
| 1 | 30 | 24 | 135 | 0 |
| 1 | 100 | 24 | 130 | 0 |
| 6 | 30 | 12 | 730 | 0 |
| 6 | 100 | 12 | 792 | 0 |

An "emergency decision" means that no stopping route passes the Eq. 7 check and no previously checked stop is still valid.
The robot then brakes to the nearest stage where it can stop, and this fallback is not checked against ψ.
In 4 more seeds with 6 obstacles and M = 30 and 100, the count was also 0.
For the cause of earlier emergencies, see [Timing of decisions](#timing-of-decisions).

### Timing of decisions

Each control decision holds for one path segment, so decisions happen at stage boundaries *inside* a control cycle.
Time-to-Arrive ψ is computed from the obstacle positions at the start of the cycle.
So a decision taken `e` seconds into the cycle uses `ψ − e` (`CycleDecider` in `src/tospt/controller.py`).

An earlier version subtracted a fixed latency of one full cycle instead.
That is safe for one decision, but inconsistent between decisions:
- between two decisions in consecutive cycles, ψ is charged a full cycle of obstacle motion;
- the robot, however, may have moved for less than one cycle.

A stop that was checked at one decision could then fail at the next one by a fraction of a millisecond, while the robot
was braking toward that stop. This produced all the emergency decisions seen before: 1 run with 1 obstacle, 38 and 27 with 6 obstacles.
I first thought that most of them came from the floor quantization of the grid routes, but fixing the timing alone removed all of them.

As a second safeguard, if the grid check finds no safe stop, the controller keeps the last stop that passed the check,
as long as the state is still inside its stoppable set. This stays safe, because two things hold:
- **The robot is never slower than the checked route.** It follows the greedy pass inside that set, which starts at least
  as fast as the grid route and takes the largest admissible step at every stage.
  This assumes that a faster state never leads to a slower next state, which holds when the stages are short.
- **Robot and obstacle times advance together.** Both arrival times and ψ decrease by the same elapsed time.

With the timing fix, this safeguard did not trigger in any run. It is covered by a unit test.

## Demo with robot dynamics

Not part of the paper. `experiments/robot_6dof_dynamics.py` runs the Exp. 3 scene with two changes:

- **Torque limits in the path constraints.** toppra's `JointTorqueConstraint` gives rows
  `tau(s) = m(s) u + c(s) x + g(s)`, with `m, c, g` from MuJoCo `mj_inverse` (armature included).
  They are added to the stage constraints, so the stoppable sets and Time-to-Reach respect them too.
  The planner uses 80% of the actuator limits (150 Nm, 28 Nm); the rest is headroom for feedback.
  The joint acceleration limit is loosened to 100 rad/s², so torque is the active limit.
  A 3 kg payload sits at the flange. Exact TOPP-RA time per leg: 0.608 s (0.510 s without the torque limits).
- **The robot is simulated, not set.** The Menagerie position servos are replaced by torque motors.
  MuJoCo integrates the dynamics with a 1 ms step. The safe path controller decides as before at 4 ms cycles.
  The decisions are taken at stage boundaries, as in [Timing of decisions](#timing-of-decisions).

**Tracking.** The controller computes the torque with inverse dynamics, for the acceleration that puts `q` exactly on
the reference after one semi-implicit Euler step (a discrete-time deadbeat law; MuJoCo's `implicitfast`
integrator is semi-implicit Euler here, because the joints have no damping).
I first used inverse dynamics feedforward plus PD. The robot then lagged behind the reference.
When the reference stopped, the robot still moved for some tens of ms to remove the last error,
up to 0.05 rad/s while touching an obstacle. Margins fixed that, but cost all productivity
(0 legs in 20 s with 6 obstacles). With the deadbeat law, the robot stops in the same physics step as the reference.

| obstacles | model used by controller / simulated payload | legs in 20 s | max joint speed in contact | max tracking error | max torque / limit | saturated physics steps |
|---|---|---|---|---|---|---|
| 1 | 3 kg / 3 kg | 30 | 3e-18 rad/s | 1e-21 rad | 0.81 | 0 |
| 6 | 3 kg / 3 kg | 17 | 4e-18 rad/s | 2e-21 rad | 0.81 | 0 |
| 1 | 3 kg / 4 kg | 30 | 4e-6 rad/s | 2e-5 rad | 0.99 | 1 |
| 6 | 3 kg / 4 kg | 17 | 8e-4 rad/s | 2e-5 rad | 1.00 | 98 |
| 1 | 3 kg / 4 kg, `--settle-time 0.01` | 29 | 4e-11 rad/s | 2e-5 rad | 1.00 | 15 |
| 6 | 3 kg / 4 kg, `--settle-time 0.01` | 14 | 5e-9 rad/s | 2e-5 rad | 1.00 | 119 |

All runs: 0 emergency decisions, joint speed within the limit, no MuJoCo contacts.

- With an exact model, tracking is exact to rounding, and the guarantee of the paper carries over to the simulated robot.
- With model error (33% heavier payload), the robot needs a few ms to settle after the reference stops.
  `--settle-time` subtracts this time from ψ, like a latency. 10 ms is enough here. `--tracking-margin` adds a distance
  for the position error, which was not needed here (2e-5 rad).
- The model error also uses up the 20% torque headroom, so the command saturates in some physics steps.
- The torque chatters near obstacles, because `u` switches between segments (the bang-bang behavior the paper mentions).
- The deadbeat law is a very high-gain feedback (1/h² = 10⁶ s⁻²). It works in simulation. On a real robot, sensor
  noise and actuator bandwidth would need a softer tracker and correspondingly larger margins.

## Fast stoppable sets

The stoppable sets `K[j, i]` need one one-step set `Q_i(K[j, i+1])` for every pair `j > i`, which is O(N²) small LPs.
For N = 500 on the UR5e path, they now take 0.12 s on one CPU process. My first version took 6.4 s, so this is about 50x faster.
For reference, the paper reports 0.20 s with 16 processes, on different hardware.
Three changes give the speed-up (`src/tospt/stoppable_sets.py`, `src/tospt/lp2d.py`):

1. **Sweep over `i` once, vectorized over `j`.** `K[j, i]` depends only on `K[j, i+1]`. So instead of one backward pass per `j`,
   a single loop runs over `i = N-1 … 0` and computes `K[:, i]` for all `j > i` in one numpy batch.
   Only N sequential steps remain.

2. **Solve each distinct target interval once.** Far from `j`, the sets stop changing, so many `j` share the same `K[j, i+1]`.
   `np.unique` finds the distinct intervals, and the results are copied back to all `j`.

3. **Reuse the stage polygon, and solve only on two lines (the main gain).**
   `Q_i([lo, hi])` is the x-range of the convex polygon `Ω_i = {(u, x) | a u + b x <= c, 0 <= x <= x_max}` cut by the strip
   `lo <= x + 2Δ u <= hi`. The minimum and maximum of x lie at a vertex of this cut polygon. Each such vertex is either
   - a vertex of `Ω_i` that lies inside the strip, or
   - a point on one of the two strip lines `x + 2Δ u = lo` or `x + 2Δ u = hi`.

   `Ω_i` does not depend on `j`. So its vertices are computed once per stage, and for each interval we only test whether they lie in the strip.
   On a strip line, substitute `u = (h − x) / (2Δ)`. Every row then becomes a bound on x alone, so the x-range on the line is a 1-D LP.
   It is solved with the same branch-free min/max solver as the Time-to-Reach step.

   Before this change, the code enumerated vertices of the full cut polygon for every interval: all row pairs (O(R²))
   with an O(R) feasibility check each, so O(R³) per interval (R ≈ 28 rows for 6 joints). Now each interval costs
   O(R) for the two lines plus O(V) for the vertex test (V = number of vertices of `Ω_i`).

`tests/test_stoppable_sets.py` checks the result against `scipy.optimize.linprog`, the closed-form braking distance of the 1-D car,
and the nesting property of Theorem 1.

## Notes on the paper

These are my readings of places where the text is ambiguous. Please check them against the original implementation.

1. **Alg. 1, line `K_{j,N} := {0}`.** Read as `K_{j,j} := {0}`, with the backward loop starting at `i = j - 1`.
2. **1-D LP (Eq. after "forward pass").** It uses `x_k = (k δv)^2` (as in Alg. 1), not `k δv`. The target set is `K_{j,i+1}`, not `K_{j,i}`.
3. **Eq. 7.** It is written as `τ_{j,i,k} − τ_{j,l,m} < ψ_j` for all `l`. That reduces to `τ_{j,i,k} < ψ_j` and ignores obstacles near
   intermediate poses, so this code uses `ψ_l` per stage (`rule="per_stage"`). The literal form is available as `rule="literal"`.
4. **Floor on ρ.** When one stage adds less than `δv` of speed, the grid route keeps the same index and cannot speed up.
   Then τ overestimates. Example: for the UR5e path with M = 30, τ from rest is 2.64 s, while exact TOPP-RA gives 0.64 s.
   This is conservative (later stops), not unsafe, but it lowers productivity when obstacles are near.
   It needs `δv` smaller than about `a Δ / v`, so a finer path grid needs a finer velocity grid.
5. **Rounding in floor/ceil.** Velocities exactly on the grid need a small tolerance. Without it, on GPU the top velocity index
   was dropped at some stages, and τ = inf spread back along the whole velocity-limit plateau.
6. **Stop at stage N − 1 is a dead end.** With `u` constant per segment, going from rest to rest over one segment takes infinite time.
   So the controller never chooses N − 1 as a stop.
7. **Scaling of Exp. 2.** The paper says time grows linearly in N and quadratically in M. Here the serial time grows about as N²
   and about linearly in M, which matches the O(N² M) count of 1-D LPs.

## Deviations from the paper

- **Baseline.** The Zanchettin et al. (2016) article was not accessible. `experiments/zanchettin.py` rebuilds it from the
  paper's description: the next velocity is maximized, with hard acceleration limits and `T_s (v + v_human) <= d − d_min`.
  A stop-at-goal bound was added so the car ends at 25 m.
- **Stoppable sets.** They are computed in one vectorized sweep instead of one backward pass per `j` over 16 processes
  (see [Fast stoppable sets](#fast-stoppable-sets)).
- **Robot scene.** UR5e (MuJoCo Menagerie) with its own capsule collision geoms, instead of OpenRAVE with enclosing spheres.
  The path is a toppra spline through two configurations and a lifted midpoint; the paper used Bi-RRT with a parabolic smoother.
  The robot is driven kinematically. `experiments/robot_6dof_dynamics.py` adds torque limits and a simulated robot
  (see [Demo with robot dynamics](#demo-with-robot-dynamics)).
- **Timing.** Decisions happen at stage boundaries within a control cycle and subtract the elapsed time from ψ.
  If no safe stop is found, the last checked stop is kept (see [Timing of decisions](#timing-of-decisions)).
- **Execution phase.** The batched GPU version of the stop-stage selection is not implemented. The CPU version is
  already within the paper's time budget, because route tracing stops after the last stage that could fail.
