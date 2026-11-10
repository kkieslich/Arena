# task_generator robots

Task-dispatch side of Arena's robot stack. This is the runtime counterpart
to [`arena_robots`](../../../../arena_robots/README.md): that package ships
per-robot config (URDFs, model_params, mappings) and the launch halves of
navigation adapters; this dir owns the Python that turns scenario-level
intent into goals sent at a specific robot through its bound adapter.

## Guides

- [Navigation adapters](../../../../arena_runtime/arena_runtime/arena_runtime/env/adapters/README.md): `Adapter` ABC, `AdapterCtx`,
  registration, how `RobotManager` binds one, dispatch and teardown flow,
  adding a new adapter. Paired with the launch-side
  [`arena_robots/launch/adapters/README.md`](../../../../arena_robots/arena_robots/launch/adapters/README.md).
- [Fleet manager](#fleet-manager) (below): `TaskModeSpec` schema,
  allocation rules, the `null` sink and `composite` fan-out.

## Package structure

Each `TM_Robots` subclass is a package:

- `__init__.py` (eager): declares `_NS` and (optionally) `_declare_schema`, then registers the mode on `ROBOTS_MODES` (a `TaskModeRegistry` from `tasks/registry.py`) with `namespace=_NS` and `schema=_declare_schema`. Imported at node startup.
- `impl.py` (lazy): contains the class body. Imported only on first activation.

Parameters live under `task.<mode>.<leaf>`. Mode names are shared across the
three axes; e.g. `task.scenario.file` is read by both `TM_Robots.scenario` and
`TM_Obstacles.scenario`, which intentionally load the same scenario file.

`reset()` computes the next episode and returns a `ResetPlan`: one `RobotReset`
per robot in `robots` (its `start_pose`, `task` and `humans`) and the
`forbidden_zones`. `EpisodeNode` submits the tasks, forbids the zones and places
the robots.

## Mid-episode assignment

`tick(ended, assigned)` runs every sim tick of a running episode after the
robots are judged. `ended` maps a robot name to the `PhaseState` that ended on
this tick, `assigned` holds the robots given a task from outside (RunTask,
SubmitTask) since the last tick, and the base class does nothing. A mode reassigns a robot from there through
`RobotManager.submit_task` (drops the robot's unfinished phases and appends the
new ones), `RobotManager.stop_task` or `move`. `explore` hands a robot whose task
just ended a fresh random goal and teleports a robot that ran into
`task.episode.timeout`, except a robot assigned from outside, which it leaves
alone until the next reset. Its `done` is always `False`. `guided` assigns from
its own goal clicks. `TM_Composite.tick` hands each sub-mode the part of `ended`
and `assigned` that belongs to its robots, runs every sub-mode even when one
raises, then raises the first error. A robot's runner state is reachable as
`ctx.robots[name].runner` (`active_state`, `outcomes`).

## Level selection

`TM_Robots` is level-agnostic: it places robots on the whole compacted map (all loaded
levels), with start and goal sampled anywhere. Which levels exist is set at load time
(`world:=name` loads all, `world:=name[0,3]` loads only 0 and 3), not per reset.

Crossing floors is handled below the task mode, in `RobotManager.submit_task`:
[`route_phases`](../../../../arena_runtime/arena_runtime/arena_runtime/env/robot_manager/elevator_routing.py) expands the explicit
`via` entries of a `GoToPhase` (elevator or stair half names), then, when the goal is on a
different level than the one the robot ends up on, the fewest-hop route over the elevators
and the stairs the robot climbs (`WorldManager.level_hops`). A tie in hop count goes to the
route with fewer elevator rides. Each ride becomes a door-aligned approach leg
(`ElevatorLegPhase`), an `ElevatorRidePhase` and a door-aligned exit leg. The ride is a
goto phase whose `target` is the boarding cabin, whose dispatch pose sits at the cabin's
back wall and whose `until` atom is `robot in <destination cabin>`, so it is met once the
robot has boarded and the elevator has delivered it, and replay judges it like any other
phase. The starting level comes from the `start` pose a task mode passes along with the
request (a reset submits before the robot is moved there), or from the robot's live pose
when none is given. The mechanism shim teleports the robot across while it waits inside
the cabin.

A stair is routable only through a boarding half (one with `to`: every stair half, the
boarding half of an escalator) and only for a robot whose mobile cap lists its kind in
`climbs` (see the [robots README](../../../../arena_robots/arena_robots/robots/README.md)).
A robot without it never routes through stairs, and a `via` naming a stair it does not
climb, or an exit-only escalator half, is an error. A stopped escalator stays routable, the
robot walks it. Each crossing becomes three phases, all in the abstract frame:

1. a `StairsLegPhase` on the axis outside the boarding mouth, facing the seam, 1 m or the
   robot radius plus 0.1 m out, met within 0.25 m once facing the seam within 0.3 rad
2. a `StairsRidePhase` with its dispatch pose past the trigger line (`STAIRS_SEAM_INSET`
   before the seam) as deep as the robot's disc fits 0.05 m short of the seam wall, at least
   0.05 m past the line, and a tolerance of half that overshoot, so a planner stopping within
   it has crossed. Its `target` is the partner half, so it arrives once the runtime has teleported
   the robot there
3. a `StairsLegPhase` on the axis outside the partner's mouth, facing out, met only with the
   robot centre 0.1 m past the mouth

Steps and treads are solid geometry every sensor sees, the robot climbs them under its own
physics and its planner gets no hint that they are passable.

## Task modes (`TM_Robots` subclasses)

`TM_Robots` ([`__init__.py`](__init__.py)) is the base: one instance drives
all robots in its scope, exposing `reset`, `tick`, `set_position`, `set_goal`,
and an async `done` flag. Shipped modes:

| Kind | File | Behavior |
| --- | --- | --- |
| `random` | [`random/`](random/) | one random reachable goal per robot per episode |
| `explore` | [`explore/`](explore/) | extends `random`; in `tick`, a robot whose task just ended gets a fresh random goal and a timed-out robot is teleported |
| `guided` | [`guided/`](guided/) | external controller drives the goal sequence. 2D Nav Goal clicks append waypoints, which show as numbered drag handles with Delete, Insert after and Clear chain menus |
| `stationary` | [`stationary/`](stationary/) | robot stays parked without goal dispatch, at `pos_x`/`pos_y`/`pos_theta` or a free pose sampled per episode |
| `scenario` | [`scenario/`](scenario/) | reads `start`/`goal` pairs from the world's scenario YAML. A robot past the `robots:` entries, or on an entry without `phases`, stays parked, and an episode with only parked robots runs until `task.episode.timeout` |
| `characterization` | [`characterization/`](characterization/) | open-loop maneuver sweep: publishes exact `cmd_vel` profiles through the robot's rated envelope (no nav goals), tags each maneuver with `characterization_phase` markers, odom stall watchdog |
| `fleet` | [`fleet/`](fleet/) | deliveries and pickups released at a fixed rate, handed to free robots, with charging trips below a battery level, see [Fleet task stream](#fleet-task-stream) |
| `null` | [`composite.py`](composite.py) | idle sink for robots unallocated by the fleet manager |
| `composite` | [`composite.py`](composite.py) | fan-out: each sub-TM sees a scoped `TaskContext` covering only its allocated robots |

Modes are registered by `Modes.TM_Robots` enum value in
[`tasks/registry.py`](../registry.py). `null` and `composite` are not in
the enum, they are only reachable via `set_tm_robots_composite`.

## Adding a TM_Robots mode

1. Create `tasks/robots/<name>/` as a package.
2. In `__init__.py`: declare `_NS = _REGISTRY_NAMESPACE("<name>")`, define `_declare_schema(node, ns)` if the mode has parameters, and register it on `ROBOTS_MODES`.
3. In `impl.py`: define the class extending `TM_Robots`; override `reset` to return a `ResetPlan`.
4. Add `<NAME> = "<name>"` to `Modes.TM_Robots` in [`modes.py`](../../modes.py).
5. `declare_mode_schemas(layer, node)` picks up your schema automatically at node init.

`reset` runs with the sim paused and fills these fields of the plan, leaving the
obstacle fields to the obstacle mode:

| Field | Meaning |
|---|---|
| `robots` | A `RobotReset` per robot. A robot left out keeps its adapter's own placement. |
| `robots[name].start_pose` | Where the robot starts. |
| `robots[name].task` | The `TaskRequest` it runs. Without one it stays parked. |
| `robots[name].humans` | Names its task may refer to. |
| `forbidden_zones` | Zones later placement keeps out of, in the map frame. |

`reset` submits nothing and moves nothing: `EpisodeNode` submits the tasks, forbids
the zones, resets the obstacle mode and places the robots, in that order. Sampling
through `get_positions_on_map` still reserves what it samples at once. In a
`composite`, each sub-mode's zones apply before the next sub-mode resets, so list
modes with declared positions first. `ResetPlan.merge` is last-one-wins: a later
plan's `RobotReset` replaces that robot's whole entry, its other fields add up.
Assign during the episode with `RobotManager.submit_task` and `move`.

## Request types

`TM_Robots` subclasses build `TaskRequest` values and hand them to
`RobotManager.submit_task`. The types live in
[`arena_simulation_setup.shared.task`](../../../../arena_simulation_setup/src/arena_simulation_setup/shared/task.py)
so scenario YAML, runtime generators, the episode record and arena_evaluation
share one definition. [`request.py`](../../../../arena_runtime/arena_runtime/arena_runtime/env/request.py) re-exports them with
`kind_of(phase) -> TaskKind`, the action kind an adapter dispatches.

- `TaskPhase`: one step. Every phase takes `on_failure`
  (`continue` | `stop_task` | `abort_episode`), `until` (clauses joined by
  ` or `, each atoms of the `conditions:` grammar joined by ` and ` with an
  optional ` for <seconds>` dwell; the phase completes once any clause has
  held, its atoms together, for its dwell, counted from arrival for a goto,
  or a structured condition node with `for:` holds, numeric comparisons and
  `some`/`each` over a field family, see
  [worlds/README.md](../../../../arena_simulation_setup/worlds/README.md)),
  `conditions` (clauses judged over the phase only)
  and `text` (an authored instruction overriding the rendered one).
- `LoopPhase(phases, until?)`: the phases repeated in order; the `until` is
  judged on every tick and, once it holds, drops the running phase and
  continues with the phase after the loop.
- `GoToPhase(pose | target, tolerance_radius?, tolerance_angle?, hold_time?, signal?)`:
  navigate to a pose, or to a zone, door, elevator, docked object or pedestrian
  by name. A zone target dispatches to a free map cell inside the zone with the
  robot's spawn clearance, at least `ZONE_GOAL_MARGIN_M` (0.3 m) inside its
  edge when the zone has such a cell, so nav2's 0.25 m goal tolerance cannot
  stop the robot outside the zone, a docked object (a world object with a `dock:`
  polygon) to such a cell within half the dock's inner radius of its deepest
  point (a planner's goal tolerance then cannot stop the robot outside the
  dock), facing the object, or its annotated front where the dock covers the
  object's own origin, a door or
  elevator to a point inside its polygon, and a
  `pose` authored next to the name (`{goto: pharmacy, pose: [4.0, 27.5, 0.0]}`)
  is kept as the dispatch pose.
  A pedestrian target dispatches to a standoff pose 0.8 m short of the
  pedestrian on the robot's side, facing them, taken from the judge's sample
  when the phase starts (`standoff_pose` in the task runner).
  Arrival on a named zone or docked object is `robot in <name>`, on a pedestrian
  `robot within tolerance_radius of <ped>`. `hold_time` is the park time
  (stationary within 5 cm and 5 degrees) before the phase counts as met. With
  a `signal` such as `arrived` the robot itself ends the phase by sending that
  signal (read from `Adapter.signal`) and is judged against the tolerance at
  that moment: within it the phase is met, elsewhere the phase fails and the
  episode is aborted as `signaled <signal> <d> m from goal`. `submit_task`
  rejects a signal the robot's adapter cannot send (`Adapter.signals`). With
  neither pose nor target the phase holds the robot where it stands once the
  reset has placed it.
- `ReachPhase`, `PlayGesturePhase`: arm phases, completed by their action
  result. A reach pose's `frame` (the base frame by default) is a robot link,
  put under the robot's TF prefix unless it already carries it.
- `TaskRequest(phases, conditions?)`: ordered phase list plus clauses judged
  over the whole request.

Unset tolerances, hold time and signal take the `task.episode.goto_pose.*` launch
parameters at submit. Phase indices are episode-wide: a second `submit_task`
appends its phases and marks the unfinished ones of the previous request
dropped. A `LoopPhase` is expanded at submit: its first iteration takes the
indices where the loop stands, every further iteration is appended at the end
of the list when the previous one ends, and the loop's `until` is judged on
every tick a loop phase is active. Once it holds, the running phase and the
rest of its iteration are marked dropped and the phase after the loop becomes
active. A loop without `until` repeats until the episode ends. Everything is judged by the task runner
([`env/robot_manager/task_runner.py`](../../../../arena_runtime/arena_runtime/arena_runtime/env/robot_manager/task_runner.py))
on the node's 30 Hz sim tick from a `Sample` of robot poses, ped positions and
semantic fields, with the same predicate code arena_evaluation replays offline
([`shared/judge.py`](../../../../arena_simulation_setup/src/arena_simulation_setup/shared/judge.py)).
The robot's progress is published as a `robot` semantic entity (`phase`, `met`,
`failed`, `dropped`, `violated`), its judged pose on `<robot_ns>/task_pose` (the
flattened world frame the judge and the compacted world share, not the env's map
frame, published once the episode's reset has landed the robot and a phase is
active), and the resolved phases in `EpisodeRecord.phases` together with
`map_poses`, the realized map-frame goto pose per phase.
[`shared/render.py`](../../../../arena_simulation_setup/src/arena_simulation_setup/shared/render.py)
renders any phase list to one instruction sentence per phase, and
[`shared/route.py`](../../../../arena_simulation_setup/src/arena_simulation_setup/shared/route.py)
words a goto as walking directions through the world's doors and openings.

## Fleet task stream

`fleet` ([`fleet/impl.py`](fleet/impl.py)) hands a stream of jobs to a fleet, configured under `task.fleet`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `homes` | | zone or docked object each robot starts and parks at, one per robot in robot order, no spot twice |
| `loading`, `locker` | | zone or docked object where a delivery loads, docked object where it unloads, a pickup the other way round; one robot at a time each |
| `rate`, `pickup_share` | 12, 0.3 | jobs released per hour of sim time, share of pickups |
| `load_s`, `unload_s` | 15, 10 | seconds held where a job loads and unloads |
| `charge_below`, `charge_until` | 30, 100 | battery percent that sends a free robot to a charger, and that frees it again; `charge_below <= charge_until` |
| `reserve` | 10 | battery percent a job must leave by its energy estimate, else the robot charges first |
| `wh_per_m` | 0.046 | Wh per metre driven, standby while driving included; 0 derives it from the robot's power profile |
| `detour` | 1.2 | route length over straight-line distance, for the energy estimate |
| `jam_window_s`, `jam_progress_m`, `jam_goal_m` | 60, 0.5, 1 | the jam rule (INFRA §8): a robot on a job leg or a charging trip that moves less than `jam_progress_m` within `jam_window_s`, further than `jam_goal_m` from its target, is jammed; `jam_window_s: 0` turns it off |
| `approach_m` | 5 | metres from the loading point or locker within which a robot takes that dock, or holds in place before it while another robot has it |
| `job_timeout`, `job_timeout_per_m` | 300, 5 | seconds a job leg may drive from its start, its waits before a busy dock apart, and a trip to a charger from its dispatch, before it is abandoned and the robot goes home: `job_timeout` plus `job_timeout_per_m` per metre of its route estimate; `job_timeout: 0` for none |

- Jobs are drawn from `RNG.stream("robots", "fleet")` alone, so a seed releases the same jobs at the same times in
  every world. Released jobs queue until a robot is free.
- **Docks.** A job has two legs, one goto per dock, each `on_failure: stop_task`. A robot drives towards its leg's
  dock without holding it; within `approach_m` of the dock it takes it if no robot has it and none waits for it, and
  holds it until it has loaded or unloaded there. Otherwise it holds in place where it is and queues for the dock;
  a freed dock goes to the robot waiting longest. So the loading point and the locker are held only for the final
  approach and the hold, not for the whole drive, and a robot waiting in the walkway before a busy dock is part of
  the traffic. A loaded robot drives straight on to its second dock. Queued jobs go to free robots oldest first,
  whether or not their first dock is busy.
- **Jobs.** Each tick, queued jobs go to free robots at or above `charge_below`, fullest battery first (a robot whose
  charge is unknown last). Before a robot takes a job the mode estimates its energy: the straight-line distance from
  the robot to the first dock, on to the second and on to the charger nearest the second, times `detour`, at
  `wh_per_m` (so the reserve is what is left once the robot could be at a charger), plus the robot's standby power
  (the power profile's `static_power_w`) over `load_s + unload_s`. When that would leave less than `reserve` percent
  of the robot's battery capacity (`task.episode.battery.capacity_wh`, else its power.yaml), the robot joins the
  charger queue instead, unless it is at `charge_until` already. The default 0.046 Wh/m is the jackal's 42.2 W
  standby plus 50 W per m/s and 17 W per rad/s at 0.4 m/s and 0.2 rad/s; `wh_per_m: 0` computes the same from each
  robot's power profile. The default `detour` 1.2 is about the 90th percentile of geodesic over straight-line distance
  on the `smu_link_full` map (mean 1.09, 0.25 m grid, 0.3 m clearance).
- **Ends.** A leg that ends failed fails its job; a leg driving longer than its timeout, or a job whose robot is
  assigned a task from outside, is abandoned. Each leg's timer starts when the leg does and stops while the robot
  waits before a busy dock, so queueing never abandons a job (the waits stay in the task time `ended - released`,
  and `job.<i>.wait_s` sums them). A leg's timeout scales with its route: `job_timeout` plus `job_timeout_per_m`
  times the straight line from where the leg starts to its dock, times `detour` (for a charging trip, to the
  charger), so a 313 m leg across the twin gets about 300 + 5 x 1.2 x 313 = 2,180 s and a short one about 360 s.
  The holder's own timeout frees a dock nobody reaches. A leg that cannot be submitted (its dock has no free cell for
  the robot) is sent again each tick until it goes out, or abandoned once its timer runs out; it never counts as done. A job that did not complete is not queued again: it ends
  there, its docks are freed and the robot goes home (or runs the outside task, and goes home once that is done).
- **Jams.** Each tick the mode keeps the positions of the robots on a job leg or a charging trip. A robot whose
  positions over the last `jam_window_s` all lie within `jam_progress_m` of the first is jammed: its job fails with
  reason `jam` (its charge is abandoned with reason `jam`), its docks are freed and it is teleported to its home,
  where it parks; a robot jammed on its way to a charger goes back to the bay it had before the trip, and the charger
  to its holder. Within `jam_goal_m` of its dock or charger it is docking or holding, and its window starts again.
  Each jam is logged with the distance to the nearest other robot, so robot-robot jams can be told apart.
- **Charging.** Free robots under `charge_below` join the charger queue and wait at their home. Queued robots go to
  a charger lowest battery first: a robot parked on its charger home charges there, any other takes a charger whose
  holder is away (on a job, or freed) and swaps homes with it. Chargers are the world's objects with
  `charge_power_w`. A robot is free again at `charge_until`.
- **Battery.** The charge is `RobotManager.soc`; the mode makes no charging decision on a robot whose charge is
  unknown. A robot whose battery reaches 0 % is logged and counted (it keeps driving: the power publisher clamps at 0).
- No zone is forbidden at the homes, since a forbidden zone stays in the map for the whole episode. The reset fails
  when a name is no goto target of the world, a home repeats or is a dock, or `charge_below > charge_until`. `done`
  is always `False`.

### Fleet task log

The mode publishes its log as a semantic entity `<env>/fleet` of kind `fleet` on the env's `state/semantics`
snapshot, which the recorder keeps (`semantic_snapshot`, one row per field). Every field is a discrete string,
written once and never changed, so the episode's last snapshot holds the whole log. Times are seconds since `t0`,
with 3 decimals; battery charges are percent with 2.

| Field | Value |
| --- | --- |
| `t0` | sim time in s the episode's job clock starts at (the reset) |
| `job.<i>.kind` | `delivery` or `pickup` |
| `job.<i>.released` | release time from the seed's stream |
| `job.<i>.robot`, `job.<i>.assigned` | robot that took the job, and when |
| `job.<i>.energy_wh` | the energy estimate it was taken on |
| `job.<i>.timeout_0`, `job.<i>.timeout_1` | seconds each leg may drive, waits for its dock apart, before the job is abandoned (0: none) |
| `job.<i>.wait_s` | seconds the robot held in place before busy docks over the job, written when it ends (part of `ended - released`) |
| `job.<i>.loaded` | when the robot finished loading |
| `job.<i>.ended`, `job.<i>.outcome` | when the job ended, as `completed`, `failed` or `abandoned` |
| `job.<i>.reason` | for `failed`, the failed phase's reason; for `abandoned`, `timeout` or `assigned from outside` |
| `charge.<n>.robot`, `charge.<n>.trigger` | robot, and why it joined the charger queue: `charge_below`, or `job <i>` for a job its battery would not last |
| `charge.<n>.queued` | when it joined the charger queue |
| `charge.<n>.charger`, `charge.<n>.dispatched` | charger it was sent to (or charges at, at home), and when |
| `charge.<n>.started`, `charge.<n>.soc_start` | when it arrived at the charger, and its charge then |
| `charge.<n>.ended`, `charge.<n>.outcome`, `charge.<n>.soc_end` | when the charge ended, as `charged` (reached `charge_until`) or `abandoned`, and the charge then |
| `charge.<n>.reason` | for `abandoned`: the failed trip's reason, `timeout` or `assigned from outside` |
| `empty.<k>.robot`, `empty.<k>.at` | a robot whose battery reached 0 %, and when |
| `jam.<k>.robot`, `jam.<k>.at`, `jam.<k>.target` | a jammed robot, when, and the dock or charger it was headed for |
| `jam.<k>.job` or `jam.<k>.charge` | the job that failed, or the charge abandoned, by the jam |
| `jam.<k>.robot_m` | the distance to the nearest other robot then (a robot-robot jam is one within a robot's length) |

`<i>` is the job's index in the stream, `<n>` and `<k>` count from 0 per episode. A job without `ended` was still
queued or running when the episode ended; a job without `robot` was never assigned. For the robot service measures:
task time is `ended - released` of completed jobs (filter on `released` for the warm-up), success rate is completed
jobs over released ones (jobs still open at the episode end apart), charger queue time is `dispatched - queued`, and
tasks per hour count completed jobs per hour of `ended`. The mode also logs each release, assignment, end, refusal, charge and empty battery at info or warning level.

## Fleet manager

[`fleet_manager.py`](fleet_manager.py) resolves episode-level
`task_modes:` entries (from scenario config) to the live
`RobotManager` instances.

### `TaskModeSpec`

```yaml
task_modes:
  - kind: random              # TM_Robots kind (or "null")
    produces: goto_pose       # default; matched against robot.accepts
    assignments: [jackal_0]   # pin specific robots by name; [] = pool
    config: { ... }           # pass-through to the TM loader
```

### Allocation (`FleetManager.match`)

1. **Pinned first.** Every `name` in each spec's `assignments` must exist in
   the robot set and must not appear on multiple specs. The spec's
   `produces` kind must be in that robot's `accepts`; otherwise error.
2. **Pool next.** Each unpinned robot joins the first unpinned spec whose
   `produces` is in the robot's `accepts`. Greedy / first-fit, specs
   earlier in the list get priority.
3. **Null sink.** If a spec with `kind == "null"` exists, every still-
   unallocated robot joins it. Without a null spec, unallocated robots
   receive no TM (they sit idle).

Result: `dict[TaskModeSpec, list[RobotManager]]`, one entry per spec, in
the input order.

### Composite wiring

[`Task.set_tm_robots_composite`](../task.py) takes the `FleetManager`
allocation and:

1. For each `(spec, robots)` pair, resolves the loader:
   `Modes.TM_Robots(spec.kind)` via the standard registry,
   falling back to `get_extra_tm_loader(spec.kind)` for sentinel kinds
   (`null`).
2. Builds a scoped `TaskContext` via `_scoped_ctx` that exposes only the
   allocated robots through `ctx.robots`.
3. Instantiates one sub-TM per spec and wraps them all in
   [`TM_Composite`](composite.py), whose `done` is `all(sub.done)` and
   whose `tick` / `set_position` / `set_goal` fan out.

Entry point: the `task.config` node param, read by
`Task.create` and at every reset. A set config takes precedence over
`task.robots`. The composite is rebound when the config path changes or the
fleet's robot names differ from the last allocation.

## Integration points

- **`RobotManager`** ([`env/robot_manager/robot_manager.py`](../../../../arena_runtime/arena_runtime/arena_runtime/env/robot_manager/robot_manager.py)):
  one per spawned robot. Owns the bound adapters, the task runner with this
  episode's phases, the goal-republish loop, and the tf-backed `pose`
  property. Entry points used from here: `submit_task`, `stop_task`, `move`,
  `is_done`, `accepts`.
- **`RobotsManager`** ([`env/robot_manager/robots_manager.py`](../../../../arena_runtime/arena_runtime/arena_runtime/env/robot_manager/robots_manager.py)):
  diff-driven fleet lifecycle. Parses the `robot` ROS param (comma-
  separated list of robot names, `model[count]`, or `.yaml` setup refs),
  reconciles against existing `RobotManager` instances, and creates /
  destroys / updates to match. Reset-time entry is `set_up`.
- **`TaskContext.robots`**: the mapping `TM_Robots` subclasses iterate;
  `TM_Composite` replaces it with a scoped view so each sub-TM only sees
  its fleet slice.
