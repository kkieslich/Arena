# FLOWEDIT graph

One graph per site with each object's demand on its node, what each object is for, and each door's access. It is a [KG](../../../../AUTHORING.md#kg) scene (`arena-kg/1`) over `core@2`, `space@1` and [`flow@1`](../../kg/packages/flow@1.toml), kept in its own `flowgraph.yaml`, never in the world's `kg.yaml`. Edits change it, a forecaster reads it, simulators never read it. Persons stay in agent types.

```
world.yaml (geometry)          --+
scenario objects (humansim:)   --+-- build --> flowgraph.yaml --+
kg.yaml (room purposes, read)  --+                              +-- read --> scenarios/<name>/ --> HumanSim
                                         crowd settings + seed -+
```

`build` turns a hand-authored scenario into its graph, `edit` changes a site and saves it as a new world, `read` turns a site back into a scenario with no hand-written person.

## CLI

`flowedit build <world> [<scenario>]`, or `python3 -m arena_simulation_setup.generative.flowedit.cli build ...`. A dry run prints the graph, `--write` writes it to `flowgraph.yaml` next to the scenario's `scenario.yaml` (or `--out`). Worlds resolve as in [generative/scenario](../scenario/README.md#cli).

`flowedit read <world> [<scenario>]` [reads](#reading-a-site) the site of a shipped world and scenario, or of a world `flowedit edit` saved, into `<world>/scenarios/flowedit_<scenario>/` (or `--name`). A dry run prints the scenario, `--write` writes it and replaces only a scenario carrying the generated marker unless `--force`. `--settings` names a crowd settings file, `--count`, `--scale` and `--local-planner` override it, `--seed` (default 0) keys every draw. `flowedit run` and `flowedit sweep` [record PEDS runs](#recording-runs).

```bash
flowedit build hospital_1 queue --write
flowedit read hospital_1 queue --settings crowd.yaml --seed 3 --write
flowedit read hospital_1__b6b7b494 --settings crowd.yaml --seed 3 --write
```

`flowedit edit <world> <type> <target> [-p KEY=VALUE ...] [--scenario <scenario>]` applies one [edit](#edits). A dry run prints the edit record and the world name, `--write` saves the world. `-p` reads VALUE as YAML and nests dotted keys (`-p satisfies.thirst=80`). An edited world is edited again by its name; its scenario and seed come from its `edits.yaml`.

```bash
flowedit edit hospital_1 move obj:fountain --scenario queue -p x=15 -p y=11.5 --write
flowedit edit hospital_1__move_fountain__b6b7b494 layout door:or_door -p op=remove --write
```

`flowedit forecast train|score|curve|scaling|fit|predict` [forecasts](#forecasting-flow) a world's flow map from its graph and geometry: it trains the learned forecaster on a split's recorded runs and scores it against the fair baselines.

`flowedit generate <world> [--scenario <scenario>] [--seed N] [--per-type K] [--steps N] [--types move,...] [--controls]` samples edits with the [generator](#generator). A dry run prints the manifest, `--write` saves every edited world and writes the manifest to `$ARENA_ASSETS_DIR_LOCAL/flowedit/<world>__seed<seed>.yaml` (or `--manifest`).

```bash
flowedit generate hospital_1 --scenario queue --seed 0 --controls --write
```

## Graph

| Node | Holds | From |
|---|---|---|
| `room:<zone>` | `has_purpose purpose:<role>` | the world KG's `has_purpose`, else the substring rules of [`zones.py`](../scenario/zones.py) as `params: {by: rule}` |
| `door:<name>` | `connects` between two rooms | derived from geometry |
| `obj:<name>` | prop `model`, `at floor:<level> {x, y, yaw, pose_ref}` | `world.yaml` objects and the scenario's `objects:` |
| `obj` demand | `serves service:<type> {capacity, interaction_radius}`, `satisfies need:<need> {amount}`, `has_formation formation:<type> {<params>, anchor_kind, anchor_ref}` | the object's `humansim:` block |
| `obj` dwell | `dwells <mean s> {std, clip_low, clip_high, from}` | the `duration` of the first agent type step whose `target` is the object's name, else its type, else of the interaction a `go_to` to it leads into |
| `obj` use | `affords interaction:<KIND> {from, meet, tag, offer, seek}`, `satisfies need:<need> {amount, from}` | [`interactions.py`](interactions.py): the agent type steps that use the object, else its `humansim:` block against HumanSim's interaction registry |
| `obj:<name>_demand` | no `model`: only `at` and the demand facts `<name>` had | a `move` with `leave_demand` (the `moved_without_demand` control); its `affords` go with it |
| `door` access | `gated <locked>`, `authorizes <name>` per authorized name, `unlocks_on regime:<regime>` | the door's `gate` preset |
| `schedule:<name>` | `asserts regime:<regime> {default}`, `window <start s> {end, value}` per window | the world's `schedules:` |

`contains` and `connects` are derived as when a world's KG loads. Unlike `kg.yaml`, the graph carries each object's `model` and pose, a copy of `world.yaml` that `flowedit build` refreshes; rooms and doors carry no geometry. Poses are in the level frame. A scenario object posed at a zone name gets the point the runtime would sample under `--seed`, and keeps the name as `pose_ref`. Dwell is read from the scenario-local agent type files (`./<type>.yaml`) its humans and regions name, and a note names other steps that give the same object another duration. `seats` and formations anchored at a pose are not in `flow@1`.

```yaml
- {subject: 'obj:fountain', verb: at, object: 'floor:0', params: {x: 11.0, y: 10.0, yaw: 1.57}}
- {subject: 'obj:fountain', verb: serves, object: 'service:fountain', params: {capacity: 1}}
- {subject: 'obj:fountain', verb: satisfies, object: 'need:thirst', params: {amount: 100.0}}
- {subject: 'obj:fountain', verb: has_formation, object: 'formation:line', params: {base_step: 1.0, front_offset: 0.8}}
- {subject: 'obj:fountain', verb: dwells, object: 5.0, params: {std: 1.5, clip_low: 4.0, clip_high: 6.0, from: 'colleague.yaml#drink.queue_and_drink'}}
```

Load and save it with `arena_simulation_setup.kg.load` and `write`.

## Interactions

`affords` says how people use an object, one fact per HumanSim `InteractionType`. A step with an object `target` and an object-bound kind (`USE`, `QUEUE_USE`, `SIT_ON`, `LIE_ON`) uses that object. A `go_to` to an object followed by a step whose kind takes no object (`GROUP_CONVERSATION`, `TALK_TO`, `HUG`, ..., or the tag-addressed `SERVICE`) uses it as a meeting point: `meet: true`, the `SERVICE` tag as `tag`, and `offer` / `seek` for the provider and seeker sides it was seen with. A need such a step satisfies becomes a `satisfies` fact with its `from`. An object with a `humansim:` block that no step uses is read by rule against HumanSim's registry: seats, or a posture word ending its type (`bed` prone, `chair`/`seat`/`sofa`/`bench`/`stool`/`couch` seated), pick the object-bound kind of that posture; else its formation picks the kind whose default formation it is, an object-bound one first (`line` is `QUEUE_USE`, `f_formation` the meeting kind `GROUP_CONVERSATION`); else a posture word of its model; else a capacity is `USE`. A block with none of these is a marker and affords nothing. `BLOCK` targets an agent and never an object.

```yaml
- {subject: 'obj:fountain', verb: affords, object: 'interaction:USE', params: {from: 'colleague.yaml#drink.queue_and_drink'}}
- {subject: 'obj:gathering_area', verb: affords, object: 'interaction:GROUP_CONVERSATION', params: {from: 'colleague.yaml#chat.hold_conversation', meet: true}}
- {subject: 'obj:patient_bed_1', verb: affords, object: 'interaction:LIE_ON', params: {from: 'rule:prone'}}
```

## Edits

[`edit.py`](edit.py) has one operation per edit type. Each changes the graph and the world's geometry together, then recomputes `contains` and `connects`. A **site** is a world without its scenario objects, plus its graph (`open_site`). `apply(site, type, target, params)` returns a new site and leaves the old one unchanged. `save(site)` writes it.

| Type | Target | Params | Graph | `world.yaml` |
|---|---|---|---|---|
| `move` | `obj:<name>` | `x`, `y`, `yaw`, `level` (default: unchanged), `leave_demand: true` (the demand stays at the old spot) | `at`; with `leave_demand` the demand facts, `affords` among them, go to a new `obj:<name>_demand` at the old spot, turned so people queue at the model's front | the pose, for a world object, its height, roll and pitch kept and its `dock`, in its own frame, and `charge_power_w` moved with it; with `leave_demand` its `humansim:` block keeps only `seats`, the dock stays with the model |
| `insert` | `obj:<new name>` | `model`, `x`, `y`, `z` (0), `yaw` (0), `level`, `dock` (where a robot stands to use it, `[[x, y], ...]` in the object frame, see [AUTHORING.md](../../../../AUTHORING.md#4-add-a-scenario)), `charge_power_w` (the power in W a robot in the dock charges with, above 0, only with a `dock`: the object is a charger), and any `function` params | entity, `at`, demand | a new world object with its `humansim:` block, `dock` and `charge_power_w` |
| `remove` | `obj:<name>` | none | the entity and every fact about it | the object with its `dock` and `charge_power_w`, and its `kg.yaml` entity |
| `function` | `obj:<name>` | `type`, `capacity`, `interaction_radius`, `satisfies: {<need>: amount or null}`, `dwell: seconds, {mean, std, clip_low, clip_high} or null`, `clear: true` (first drop all demand, the formation included) | `serves`, `satisfies`, `dwells` (`from: edit:<id>`), `has_formation` (by `clear`) | the `humansim:` block, for a world object |
| `access` | `door:<name>` | `locked`, `authorized`, `unlock_on`, or `remove: true` | access facts | the door's `gate` preset (new gates start unlocked) |
| `access` | `schedule:<name>` | `regime`, `windows`, `default`, or `remove: true` | access facts | the `schedules:` entry |
| `layout` | `door:<name>` | `op: add` (`level`, `span`, door fields), `move` (default: `span` or `offset: [dx, dy]`, door fields), `remove` | the door entity | the door |
| `layout` | `wall:<level>/<index>`, `wall:<level>` to add | `op: add` (`path`, `closed`, `kind`, `material`), `move` (`path` or `offset`), `remove` | none | the `walls:` path at that index |

- **Scenario objects** (the fountain of `queue`) change only in the graph. The [reader](#reading-a-site) writes them into the scenario.
- **Demand-only nodes** (`obj:<name>_demand`, an object without `model`) also live only in the graph. The reader writes each as a PEDS target without an obstacle (`model: ""`). The runtime queues an object without an annotated `face` behind its yaw, and a node without a model has none, so the node is turned to put that side where its model's front was: people queue on the same side as before the move. A formation anchored at `<name>` by name is anchored at the new node. They can be moved, removed and given another function like any graph object; the generator never moves or copies them, having no model.
- **Doors:** a moved or added door must lie on a wall. A door moved or removed from a gap left in its wall gets the gap closed with a wall styled like the wall at its start. Walls drawn through a door are cut at runtime anyway.
- **Walls:** only `walls:` paths count. Walls from a zone's `walls: outline` are not addressable.
- **Checks** stay basic: unknown target, taken name, door off a wall, then `World.validate` and the graph's vocabulary. Rules about rooms or overlap belong to the edit generator.

`save` writes the world to the writable local root (`$ARENA_ASSETS_DIR_LOCAL/worlds`, see [worlds/README.md](../../../../worlds/README.md#how-worldidentifier-resolves-a-name)) as `<base>__<what>__<id>`, e.g. `hospital_1__move_fountain__9f4189dc`. `<what>` says what the edits did: the last edit's type, layout `op`, target name and switched-on flags, after `<n>_edits` for a chain (`describe`, at most 40 characters). `<id>` is a hash of the last edit's record, with ints read as floats and false flags (`leave_demand`, `clear`, `remove`) left out. The record names its parent by the parent's key `<base>__<id>` (`Site.key`), so the hash also names the whole chain and the readable part never enters it. The save hardlinks the files of the source world's `assets/` (copies them where a link fails, e.g. across filesystems), so a copy of a 1.4 GB twin costs no space; an asset of an edited world is shared with its source and is replaced, never written in place. It writes `world.yaml`, `flowgraph.yaml` and `edits.yaml` as its own files. The scenarios are not copied.

```yaml
format: flowedit-edits/1
base: hospital_1
scenario: queue
seed: 0
edits:
- {id: b6b7b494, parent: hospital_1, type: move, target: 'obj:fountain', params: {x: 15, y: 11.5}}
- {id: b9f2feba, parent: hospital_1__b6b7b494, type: layout, target: 'door:or_door', params: {op: remove}}
```

## Generator

[`generate.py`](generate.py) samples edits of every type on a site from a seed. `generate(site, seed, per_type, steps, types)` makes, per type, `per_type` chains of up to `steps` edits (default one), each starting with an edit of that type; later steps draw their type from `types`, and a drawn type with no edit (a function edit once the only object with a need is gone) gives way to the others. Each step draws a proposal, `apply` checks it, and a refused one is drawn again, up to `MAX_TRIES`. A site with nothing to edit, or no valid proposal, gives a note instead, which names the last refusal. A chain whose first edit fails is left out; one whose later step finds no edit of any type ends there.

- **Determinism:** one `random.Random` per step, keyed by the current world's key (`<base>__<id>`, which names the chain so far), the scenario, seed, step, type and index. Candidates are sorted before every draw and spots rounded to cm, so the same site, seed and code give the same edits and a shorter chain is a prefix of a longer one. No two chains reach the same world at the same step: a duplicate is drawn again, and that retry may take any object, not only one with a need. Footprints come from asset bounds where they resolve, else a 0.8 m box with its front behind the yaw, so a machine with other assets at hand can draw other spots; the manifest carries `generator_version`. The model of an object with a `dock` must resolve: its fallback front could turn the dock to the wall, so `resolve_footprints` refuses it with the model's name (a generated move or insert on its level, and a `move` with `leave_demand` of it), and with `strict=True` it refuses any model that does not resolve.
- **On the floor:** spots are drawn on the floor, so a world object standing higher than 5 cm (on a table, a shelf or a wall) is neither moved nor copied.
- **Spots** (`move`, `insert`): inside an allowed zone (not a stair, elevator, lift or shaft), the footprint clear of the zone's edge, walls, door spans (0.8 m) and, by 0.3 m, every object's footprint. A moved object's own old footprint counts too, so the inert copy of its control never overlaps the original. Every spot stands against a wall of its zone: it is drawn along the parts of the zone's edges a wall runs along (within 0.3 m of the edge's line and 10° of its direction, parts shorter than 0.6 m skipped, so a door gap only cuts its edge and a wall stub or a wall meeting the edge gives none), as close as those gaps allow, with a wall behind the middle and both ends of the object's back, and the model is turned so its annotated front (`face` in its annotation.yaml, read as the runtime reads it; without one the side behind its yaw, where the runtime queues it) points away from that wall into the room; a local move draws only along the walls within reach.

| Type | Target | Params drawn |
|---|---|---|
| `move` | an object on the floor, one with a need (`satisfies`) first | a spot in its own room 0.5-3 m away (`scope: local`, 30 %), else in another room of its level (`scope: room`) |
| `insert` | `obj:gen_<model>_<n>`, a copy of an object on the floor, one with a need first | a spot in any room of its level (`scope: room`), the source's demand (service, needs, dwell), `dock` and `charge_power_w` copied |
| `remove` | an object, one with a need first | none |
| `function` | an object with a need | `capacity`: a new value in the type's range (`CAPACITY`), objects whose type others share first, none with capacity 0. Or `dwell`: a new mean in the type's range (`DWELL_S`, 3-20 s for dispensers), the authored spread scaled to it, only on the sole object for its need |
| `access` | a door whose rooms keep another door, a gated door, or a schedule with a regime | lock a door; unlock or ungate a gated one; a schedule window |
| `layout` | a door | move it 0.5-2 m along its wall; add a 1.5 m door on a wall between two rooms; remove one whose rooms keep another door |

Only objects with a need count as demand: in PEDS only needs send people to an object. `dwell` reaches PEDS through the reader, which writes it as the `duration` of the step at the object. The reader takes one dwell per need, hence the sole-object rule. `satisfies` is not drawn yet. `FIELDS` lists what is drawn and goes into the manifest. Wall paths are not drawn.

**Controls** (`controls(site, chain)`, `--controls`): a function-carrying move or insert gets geometry-only controls, applied to the chain's base, in which only the footprint changes:

| Control | Edit | Result |
|---|---|---|
| `inert_object` | `insert obj:<name>_inert` (move or insert) | the same model, `dock` and `charge_power_w` without demand at the edit's spot, the original left in place |
| `moved_without_demand` | `move obj:<name>` with `leave_demand: true` (move only) | the model at the edit's spot, its demand at the old spot on `obj:<name>_demand` |

Only a chain's first edit has controls. An object without `humansim.type` takes its type from its asset's `annotation.yaml` at runtime, so the inert copy, and the moved model of `moved_without_demand`, may still be a target by that name.

The manifest (`flowedit-generated/1`) names each chain's world, its edits (`type`, `target`, `id`, `function_carrying`, `scope`), its saved path and its controls, each with the `ids` of its own edits (`edits.yaml` keeps the full history):

```yaml
format: flowedit-generated/1
generator_version: 5
base: hospital_1
scenario: queue
seed: 0
per_type: 1
steps: 1
fields: [capacity, dwell]
chains:
- world: hospital_1__move_fountain__1a2b3c4d
  edits:
  - {type: move, target: 'obj:fountain', id: 1a2b3c4d, function_carrying: true, scope: room}
  path: /…/worlds/hospital_1__move_fountain__1a2b3c4d
  controls:
  - {kind: inert_object, world: hospital_1__insert_fountain_inert__5e6f7a8b, ids: [5e6f7a8b], path: /…/worlds/hospital_1__insert_fountain_inert__5e6f7a8b}
  - {kind: moved_without_demand, world: hospital_1__move_fountain_leave_demand__9c0d1e2f, ids: [9c0d1e2f], path: /…/worlds/hospital_1__move_fountain_leave_demand__9c0d1e2f}
```

## Reading a site

[`read.py`](read.py) reads a site from `open_site` (a shipped world and scenario, or a world `flowedit edit` saved), crowd settings and a seed into a scenario in the site's world, deterministic in all three. FLOWEDIT compares a run before an edit with a run after it, so the two may differ only by the edit:

- **The people** (count, kind, speed, start room, route, idle points) come from the unedited base world (`site.base`), its graph, the settings and the seed alone, through populate's public `crowd`. Every edit of `hospital_1` gets the people of `hospital_1`. Each start is a point drawn in the base world, not the home room's name, which the runtime would draw again per world. A start an edit made unfree (an inserted or moved object's footprint, a new wall) moves to the nearest free point of the edited world, with a note; every other person starts where they start in the base.
- **The binding** (objects, which object serves which need, the dwell, the entrances) comes from the edited site.

| Site | Scenario |
|---|---|
| objects the world does not hold (`at`, `serves`, `has_formation`, `satisfies`) | `objects:` with their `humansim:` block at the graph's point, in the map frame. A world object whose graph pose differs from `world.yaml` is refused: a scenario cannot move it, `flowedit edit` moves both |
| an object without `model` (`obj:<name>_demand`) | an object with `model: ""`: a HumanSim target with its demand, no obstacle in the simulator and none on the sampling grid |
| `satisfies need:<n>` of an object people can use | every person's type declares `n` and preempts into `seek_<n>` once it falls below `below` |
| `affords` | the `seek_<n>` step: the kind at the object, or with `meet` a `go_to` to it and the kind beside it. An object without `affords` is used as the objects of its service type are, else by rule. `SERVICE` needs a provider and is not sought |
| `dwells` | that step's `duration`, its `patience` shifted by the dwell's upper bound so a use never runs out mid-use |
| `has_purpose` of the base world's rooms | where people start and the rooms they walk between, by populate's rules |
| a door only one room contains | `<door>_source` behind `<door>_sink` inside it: visitors enter, walk, use, and leave through the sink |

People are HumanSim behavior trees: each gets its own type (`types/flow__<name>.yaml`, extending its populate mix type) whose `idle` sequence walks free points drawn in its rooms of the base world, since a tree does not follow scenario waypoints. One step serves a need (HumanSim holds the dwell on the step), so the objects that satisfy one need must share their kind and dwell, and several of them are sought by a service type that they, and nothing else on the floor, have (HumanSim takes the nearest); anything else is an error until HumanSim holds the dwell on the object. With `synthesize_needs` (off by default), an object that satisfies no need gets `visit_<type>`, so every usable object is used; in HumanSim a crowd seeking a capacity-1 seat waits where it stands, so use it with few people or low rates. There is no robot unless the settings name one (`robot:`). People walk the first level with zones. `provenance.yaml` holds the world, its base and edits, the graph hash, the full settings, the plan of every need and the notes.

```bash
flowedit edit hospital_1 function obj:fountain --scenario queue -p dwell=20 --write   # saves hospital_1__<id>
flowedit read hospital_1 queue --write                  # hospital_1/scenarios/flowedit_queue/
flowedit read hospital_1__<id> --write                  # the same people, the fountain's step 20 s
```

## Recording runs

`flowedit run <world> [<scenario>] [--edit TYPE TARGET [-p KEY=VALUE ...]]` is the whole chain in one command: apply the edit (or none) and save the world, read the crowd into it under `--settings` and `--seed` as `scenarios/flowedit_<scenario>_<settings digest>_s<seed>/`, launch it headless (`sim:=dummy human:=arena`) with HumanSim recording (`humansim.record:=true`), stop after `--duration` seconds (default 300) and write the run to `$ARENA_DATA_DIR/flowedit_runs/<world>__<scenario>/` (or `--out`). `--launch-arg` passes more `arena launch` arguments, `--dry-run` prints the world and launch without saving or launching. A finished run is skipped unless `--again`. A run is a look at one world, not one side of a pair: `--seed` keys the reader's draws, but the runtime seeds its episode from a `run_seed` drawn at random per launch, so a base and its edit run this way never get the same crowd (`paired: false` in `run.yaml`). Paired runs go through arena_evaluation's benchmark runner, which gives each stage its seed ([`flowedit suite`](#paired-runs)).

```bash
flowedit run hospital_1 queue --edit function obj:fountain -p dwell=20 --settings crowd.yaml --seed 3
```

| File | Content |
|---|---|
| `run.yaml` | world, base, edits, settings and digest, seed, launch command, the code that ran (`code`, `humansim`, each with a `_dirty` flag for changed tracked files, and `runtime_code` when `$ARENA_DIR` is another checkout), the reader's provenance, a track summary, and `uses` per object (interactions that reached their hold, their users, the longest queue); written last |
| `tracks.parquet` | HumanSim's agent states (`time` from the first frame, `agent_id`, `x`, `y`, `vx`, `vy`, `radius`, `planner`, `source_name`, `source_seq`) with `name` and `key` |
| `interactions.parquet` | HumanSim's interactions on the tracks' time base: a `member` row per member of a live interaction per frame (`type`, `outcome`, `object_id`, `arrived`, `holding`, `duration`, `role`, `position`, `seat`), an `event` row per participant of an edge (`activated`, `hold_onset`, `released`, `interrupted`, `canceled`), members keyed like the tracks |
| `humansim/` | HumanSim's recording: the bag and `recording.yaml` |
| `launch.log` | the launch's output |

`key` pairs an agent across runs: `<source_name>#<source_seq>` for a person an entrance released, never the agent id (HumanSim keys a released agent's draws on its source and release index), and the person's scenario name for the people the reader places, who are the same in every edit of a world. `name` is the agent's name at that frame: HumanSim hands a left agent's id to the next one. Tracks and interactions are cut to the first `--duration` seconds; `run.yaml` warns when the sim ran too slowly to fill them. Needs are not in the recording (HumanSim records no `viz_state`), so a use of an object is read from `interactions.parquet`, not from a need rising.

The scenario a run reads into a shipped world, `scenarios/flowedit_<scenario>_<settings digest>_s<seed>/`, is ignored by git: a sweep writes it again.

`flowedit sweep <split> [--partition P] [--settings S] [--jobs N]` records every run of a split: it generates the split's edited worlds (and controls) again, checks the generator still makes every name the split lists, then runs each world of the chosen partitions once per seed of its partition, `--jobs` launches at a time, each on its own ROS domain from `--domain-base` (default 150). Runs land in `$ARENA_DATA_DIR/flowedit_runs/<split name>/`, one log per run in `_sweep/`; finished runs are skipped, so a sweep resumes. A split file ([`split.py`](split.py)) names the worlds of `train`, `held_out_edits` (other edits of the train worlds) and `held_out_worlds` (worlds with no run in train), and per base world the `flowedit generate` call that makes its edited worlds again under those names (`controls: true` makes the geometry-only controls too). `seeds` is one list for every partition or a list per partition. `settings` (optional) holds the crowd settings of every run, a sweep's `--settings` file merged over them; the sweep writes the merged settings to `_sweep/` and hands that file to every run. `control_worlds` links each function-carrying edit to its controls, which sit in the edit's partition. Every edited world is scored against its base world run with the same seed: a partition that lists edits but not their base (held-out edits at test seeds) runs the base world at its seeds as a reference run, never trained on (`reference` in `--dry-run`, `(reference)` in the log). [`splits/peds_pilot.yaml`](splits/peds_pilot.yaml) is the pilot: hospital_1 with 3 chains per edit type and office_1 with 1, their controls, seeds 1-5 for train and 6-8 for test, 155 runs; the study design's split is larger. A sweep launches `flowedit run`, so its runs are not seed-paired either; paired runs of a split come from a benchmark suite with each stage's seed ([`flowedit suite`](#paired-runs)).

```bash
flowedit sweep splits/peds_pilot.yaml --jobs 2                                # the base crowd
flowedit sweep splits/peds_pilot.yaml --crowd test_orca --jobs 2              # one test crowd
flowedit sweep splits/peds_pilot.yaml --partition held_out_edits --settings test_crowd.yaml
```

### Test crowds

A split's own `settings` are its base crowd, which runs on every partition. `crowds` (optional) adds test crowds: each names a settings file, relative to the split and merged over the split's `settings`, and the partitions it runs on. A test crowd's run of a held-out edit comes with its base world's reference run under the same crowd, so the pair still differs only by the edit. `--crowd NAME` (repeatable, `all` for every crowd) picks the crowds a sweep records, `base` by default. Each crowd's runs carry its settings digest in their scenario name, so the crowds' runs of one world never collide.

[`splits/peds_pilot.yaml`](splits/peds_pilot.yaml)'s test crowds, in [`splits/crowds/`](splits/crowds/), run on the held-out edits at seeds 6-8 (30 runs each, 3 of them reference runs). Each differs from the base crowd in one way, `test_all` in all three:

| Crowd | Need decay mean | People at start | Local planner |
|---|---|---|---|
| `base` | 0.5 /s | populate's count | `sfm` (HumanSim's default) |
| `test_rates` | 1.5 /s (std 0.4, clipped 0.8-3.0) | populate's count | `sfm` |
| `test_size` | 0.5 /s | twice populate's count | `sfm` |
| `test_orca` | 0.5 /s | populate's count | `orca` |
| `test_hsfm` | 0.5 /s | populate's count | `hsfm` |
| `test_all` | 1.5 /s | twice populate's count | `orca` |

The reader writes the crowd's planner on every person's agent type, so the planner needs no launch argument. Entrance visitors are released by HumanSim's sources with their visitor type, which carries the same planner.

Entrance visitors walk in from the source strip (0.8 to 1.8 m inside the door) past the door's sink strip (0.1 to 0.6 m), and leave by walking into the sink. HumanSim's sink takes a released agent only while it heads for that sink (kkieslich/humansim#2), so a visitor crossing it on the way in stays. The 9 Nov pilot, [`splits/peds_pilot.yaml`](splits/peds_pilot.yaml), ran before that fix without entrances (`settings: {entrances: {rate: 0}}`), so only the reader's placed people walked; [`splits/peds_v2.yaml`](splits/peds_v2.yaml) runs with them.

### Data v2

[`splits/peds_v2.yaml`](splits/peds_v2.yaml) is the Stage 1 dataset (study design §5, decided 10 Nov): train hospital_1 and office_1, base worlds at seeds 1-5 and 8 of 10 chains per edit type with their controls at seeds 1-2 (`seeds.train: {bases, edits}`); held-out edits, the other 2 chains per type and train world drawn by seed among those the pilot never ran, at seeds 6-8; held-out worlds reception and hallway_clinic, each with an authored `demand` scenario, base and chains 1-2 per type at seeds 6-8. Each held-out world's base also runs at seeds 1-5 (`baseline_seeds`), only for the stale and transplanted baselines and the across-seed floor: those runs belong to `held_out_worlds`, outside its seeds, so nothing trains or scores on them. The test crowds run on the first held-out chain of every edit type per train world, with their controls, plus baseline runs of their bases at seeds 1-3. Door entrances are on (the defaults of `entrances`).

hallway_clinic is a generated layout (world generator `hallway`, seed 7, recorded in its `generated_from.yaml`) authored into a clinic: room names and purposes, a front and a back door, furniture and the `demand` scenario. reception got its `demand` scenario and two `has_purpose` facts (its zone names read as transit, so nobody would start there); with one door between its two rooms and none outside, it has no entrances and no access edits.

## Paired runs

`flowedit suite <split> [--partition P] [--crowd C] [--settings S] [--out FILE] [--robot R]` writes a split as an arena_evaluation benchmark suite whose runs pair by seed. It generates the split's edited worlds again with the same manifest check as a sweep, reads the crowd into every world at every seed of its partition (the reference runs included), and writes one stage per world and seed. Each stage runs one episode, its `seed` the runtime's episode seed and the reader's seed both, so an edit stage and its base world's stage at that seed run the same people on the same seed. The suite lands in `<imported assets root>/suites/flowedit_<split name>.yaml`, where `--suite flowedit_<split name>` finds it, so a run of it resumes by name. A test crowd (`--crowd C`) is its own suite, `flowedit_<split name>__<crowd>`, of the partitions it runs on: stage names do not say the crowd. `flowedit suite` prints the call:

```bash
flowedit suite splits/peds_pilot.yaml
arena evaluation benchmark --suite flowedit_peds_pilot --contest '[{name: peds_only, mobile: none}]' --lanes 2
flowedit suite splits/peds_pilot.yaml --crowd test_orca
arena evaluation benchmark --suite flowedit_peds_pilot__test_orca --contest '[{name: peds_only, mobile: none}]' --lanes 2
```

| Stage field | Value |
|---|---|
| `name` | `<world>__s<seed>`; an edit stage pairs with `<base world>__s<seed>` (`suite.pairs`) |
| `map`, `seed`, `episodes` | the world, the seed, 1 |
| `config.scenario.file` | the reader's scenario of that world, settings and seed, `flowedit_<scenario>_<settings digest>_s<seed>` |
| `timeout` | the split's `duration`, sim seconds |
| `robot`, `modes.robots`, `config.stationary` | the runner needs a robot in every stage: `--robot` (default `jackal`) parked at (1000, 1000) by the `stationary` mode, as the runner parks its `unhindered_peds` reference runs |

The suite's `launch` runs flatsim headless under lockstep with `human.robots.visible: false`, so the people never see the parked robot, and `references: false`, since the split names its own reference runs. The contestant needs no planner: `[{name: peds_only, mobile: none}]` brings the parked robot up with the no-op mobile adapter, not the sim's default navstack (Nav2 in flatsim, about 12 % of a core and 0.4 GB per lane for a robot that never moves, measured on hospital_1); the runner then expects no planner beats from it. `--dry-run` prints the suite without writing a scenario or the suite, and a suite file `flowedit suite` did not write is kept unless `--force`.

### Planners through the edit

A split with a `robot_task` (a scenario `robots:` entry: `start` and `phases`) has the robot drive it in every stage. The task is added to the scenario after the crowd is read, so the people are the ones of the robot-free suite at the same seed. The scenario is `flowedit_<scenario>_<settings digest>_r<task digest>_s<seed>`, beside the robot-free one, and the stage runs `modes.robots: scenario` instead of the parking spot, under `human.robots.visible: true`. Every goto of the task must be a pose: a named target (a zone, a door, an object's dock) can resolve differently in an edited world, a pose is the same in all of them. A loop without `until` repeats until the episode's `timeout`, so every stage drives for the split's `duration`. `contest` lists the planners, as arena_evaluation contestants; `flowedit suite` writes them as contest `flowedit_<split name>` into `<imported assets root>/contests/` and prints `--contest flowedit_<split name>`. `flowedit sweep` refuses a split with a robot task, since `flowedit run` records without one.

[`splits/h2_planners.yaml`](splits/h2_planners.yaml) is the Stage 1 test of H2: hospital_1, its first fountain move and the move's two geometry-only controls, at seeds 1-5 on data v2's base crowd (door entrances on, the same people as v2's train runs), with Nav2 DWB (`dwb_blind`), DWB with `arena_social_cost_layer` (`dwb_social`) and DRL-VO (`drlvo`) shuttling between the central hallway and the laboratory past both fountain spots: 60 episodes of 300 s. The H2 test is the contrast edit minus control per seed, for each control and planner (`evaluation edits --split splits/h2_planners.yaml`, paired t-test at n = 5, Holm within the contrasts), with edit minus base and control minus base reported beside it.

```sh
flowedit suite splits/h2_planners.yaml
arena evaluation benchmark --suite flowedit_h2_planners --contest flowedit_h2_planners --lanes 2
```

## Crowd settings

Global, not per site: the same file gives a crowd on any graph. Every key and its default is in [`settings.py`](settings.py); a file sets only what it changes, an unknown key is an error. A distribution (`mean`, `std`, `clip_low`, `clip_high`) given in a file replaces the default one whole, so a raised mean never keeps the default's clips, and a mean outside its own clips is an error. A need's own `initial`, `decay_rate` or `below` replaces the `"*"` entry's.

```yaml
count: 20                       # people at the start (default: populate's count for the rooms, times scale)
needs:
  "*": {decay_rate: {mean: 0.5, std: 0.15, clip_low: 0.1, clip_high: 2.0}, below: 30.0}
  thirst: {decay_rate: {mean: 2.0, std: 0.5, clip_low: 1.0, clip_high: 4.0}}
local_planner: orca
entrances: {rate: 0.1, max_concurrent: 6, max_total: 30}
```

## Forecasting flow

[`forecast/`](forecast/) forecasts a site's flow map from its edited graph and geometry, without a PEDS run and without any recorded flow of the site. The map is arena_evaluation's (`processing/flow/grid.py`): pedestrian-seconds per 0.5 m cell over the base world's rooms, 8 headings and a standing bin. A world's truth is its seed-pooled map (its runs' normalized maps averaged over the partition's seeds); a score is the JSD in bits against it. [`maps.py`](forecast/maps.py) mirrors arena_evaluation's grid, track binning, `jsd` and `pooled`, and tests check they agree.

Two forecasters read the same inputs:
- the **learned forecaster** (H1's forecaster): a small GNN over the graph and a U-Net on the flow grid, trained on single runs;
- the **demand model**: six parameters fitted on the train partition, kept as a baseline.

### Learned forecaster

**Input** ([`raster.py`](forecast/raster.py)), all on the base world's flow grid:
- **geometry:** the rooms, the free space people walk (HumanSim's 0.38 m inflation, 0.25 m sub-cells), its clearance, open doors, locked doors (`gated true`), entrances (doors in one room only, weighted by the entrance rate);
- **room purpose:** one layer per purpose (`work`, `break`, `transit`, `private`, `unknown`);
- **people:** the reader's start points and idle stops of the unedited base world, at the 16 crowd seeds no split uses (1000–1015), each person weighted 1/16;
- **per need** the graph's objects satisfy:
  - the objects' fronts, weighted by capacity and by dwell;
  - their queue spots;
  - the walking distance to the nearest front, as exp(−d / 10 m);
  - the need's expected trips per person over the run;
- **the graph:** nodes for rooms (purpose, area, people, doors), doors (open, locked, exterior, width), demand objects (capacity, dwell, front offset, queue step) and needs (trips, amount, objects). Typed edges join room–door, object–room, object–need, room–need and object–object of one need. Each edge carries the walking distance between the two anchors.

**Network** ([`net.py`](forecast/net.py)):
1. A typed GNN, 3 rounds of message passing with 16-dimensional embeddings, lets objects that compete for one need, their capacities and the rooms people start in meet.
2. The room, door and object embeddings are written onto the room's cells, the door's span and the object's front and queue spots.
3. A per-need encoder (shared convolutions, summed over needs) takes any number of needs.
4. A U-Net (32–64–96–128 channels) maps everything to 9 non-negative rates per cell. The rates are scaled by the crowd's pedestrian-seconds per room cell and kept to the rooms. Being fully convolutional, it takes any world size.

**Training** ([`learn.py`](forecast/learn.py)):
- Every single run of the train partition is a sample.
- Loss: Poisson NLL of the run's pedestrian-seconds per cell and bin (as quasi-likelihood counts), plus half of the same on 2 × 2 blocks (1 m cells). So the network learns the expected flow from noisy single seeds.
- Augmentation: the eight D4 symmetries (quarter turns, mirror), with the heading bins permuted along.
- Optimizer: AdamW with a one-cycle schedule, 3000 steps of 8 runs.
- A forecast is the mean over the eight symmetries.
- `--no-demand` drops the need channels and the object and need nodes: the **demand ablation**, the same network otherwise.

**Validity** ([`validity.py`](forecast/validity.py)):
- `score`: each forecaster and the fair baselines per world of a partition, against the across-seed floor.
  - **stale** is the base world's pooled map from its runs at the split's *other* seeds, never the scored crowd draws. **transplanted** is that map with the edited object's surroundings (2 m) moved, copied or dropped with it. A partition whose base world ran at no other seed (the pilot's office_1) has neither.
  - **floor** is the base world at the scored seeds against itself at the other seeds: what a new crowd draw of an unchanged world costs.
  - `forecaster 1m` is the occupancy JSD on 1 m cells: the coarse scale at which the Bayesian Floor Field transferred to unseen worlds.
  - A second table puts each edit's and control's **recorded change** (against its base at the same seeds) beside each forecaster's **predicted change** (its map of the edit against its map of the base). The inert copy and the object moved without its demand should change little, the function-carrying edit a lot.
- `curve`: forecast error against post-edit tracks seen.
  - The forecast is a Dirichlet prior over cells and bins with strength c (pedestrian-seconds, fitted on the train partition). After k tracks the map is (c·p + n) / (c + N), the Bayesian Floor Field's conjugate update.
  - Leave-one-seed-out: the tracks come from one scored seed's run, and the score is against the pooled map of the partition's other seeds, averaged over which seed is seen and over 20 random track orders.
  - Zero tracks is the forecast itself (the flat line). The same update from a uniform prior over the rooms is the online-only reference.
- `scaling`: the learned forecaster trained on 25/50/100 % of the training edit chains (an edit with its controls; the base world always kept), 3 draws each, scored on the held-out partitions.

```bash
S=splits/peds_pilot.yaml; C=<benchmark dir>/flow_cells.parquet
flowedit forecast train $S --cells $C --out net.pt
flowedit forecast train $S --cells $C --no-demand --out geometry.pt
flowedit forecast fit $S --cells $C --out params.yaml
flowedit forecast score $S --cells $C --net net.pt --net geometry.pt --params params.yaml --predicted cliff=cliff.parquet --plots plots/ --out score.yaml
flowedit forecast curve $S --benchmark <benchmark dir> --net net.pt --png curve.png --out curve.yaml
flowedit forecast scaling $S --cells $C --out-dir scaling/
flowedit forecast predict office_1__move_fountain__2af43765 --net net.pt --split $S --png forecast.png --out forecast.parquet
```

Training takes a few minutes on a GPU (`--device`, default cuda when available).

### Map files (`score --predicted`)

Any forecaster or baseline joins `score` as a parquet file of cell rows, one map per world, on the world's base grid:

| Column | Type | Meaning |
|---|---|---|
| `flow_world` | string | the world (split name), e.g. `hospital_1__move_fountain__6a2c296b` |
| `x`, `y` | float | the cell centre in the world frame, as arena_evaluation's `flow_cells.parquet` has it (`x0 + (j + 0.5) · 0.5`) |
| `heading_rad` | float or null | the heading bin's centre, −π + (b + 0.5) · π/4 for b = 0…7; null for standing (under 0.2 m/s) |
| `ped_seconds` | float | the map's weight in that cell and bin; only its distribution over cells and bins counts, not its total |

- Omit cells without weight.
- A world missing from the file scores as "no map".
- `forecast predict --out` and arena_evaluation's `flow_cells.parquet` (its `flow_world` column) are in this format.
- `validity.write_predicted` writes one from numpy maps (ny, nx, 9).
- Baselines that read recorded runs must take them from seeds other than the scored ones (`validity.other_seeds`). For held-out edits, those are the base world's train seeds.

### Demand model

[`model.py`](forecast/model.py) mirrors the [reader](#reading-a-site) in expectation:
- **People** are populate's crowd of the unedited base world, as the reader draws them (start point, idle stops, speed), at the 16 crowd seeds, each person weighted 1/16. A person walks from its start to its first stop, then cycles through its stops, standing `idle.stand` at each (the reader's `idle` sequence).
- **Needs** send people to the objects the edited graph says satisfy them (the reader's `plan_needs`). A need starts at its initial value, decays at its rate, and starts a trip below `below`; a use adds `amount`. The expected number of trips within the run's duration comes from the settings' distributions. A trip leaves from one of the person's stops, walks to the object's front, waits, uses the object for its `dwells`, and walks back to the first stop, where `idle` starts again. The front lies `front_offset` from the object along its annotated `face`, else behind its yaw, as the runtime queues it.
- **Choice and queues.** The objects of one need share a person's trips by a softmin of their path lengths with temperature τ (HumanSim takes the nearest). Each object's arrival rate λ gives its utilisation ρ = λ · dwell / capacity (at most 0.9) and a wait q · dwell · ρ^c / (2c(1 − ρ)), an M/D/c-like wait with c = capacity. The wait is stood on the queue's spots, `base_step` apart behind the front, in proportion to ρ^k.
- **Walks** follow shortest paths on the edited world's free space: 0.25 m cells that keep HumanSim's navmesh inflation (0.38 m) from walls and objects, its comfort radius (0.6 m) as a cost, and locked doors (`gated true`) closed. Walking time goes to each step's heading, averaged over 5 steps and spread by σ_θ over the heading bins. Standing time goes to the standing bin. A person whose start is off the free space or cut off from its stops stands there for the whole run.
- **Map.** The time is blurred by a Gaussian of width σ, kept to the reachable cells of the rooms, and scaled to people × duration.

Six parameters are fitted on the train partition (`forecast fit`): σ (kernel, m), σ_θ (heading spread, rad), w (walking mass per walked second, relative to standing), g (trips per expected trip of the needs model), τ (softmin temperature, m) and q (queue wait gain). Nelder–Mead in log space minimises the mean pooled JSD over the partition's worlds. `fit --crowd runs` draws the people at the fitted runs' seeds: a diagnostic, not a forecaster.

**Neither forecaster models yet:**
- entrance visitors (the pilot runs without them; the learned forecaster reads the entrances but has seen none);
- schedules and regimes;
- what `meet` and `SERVICE` interactions do beyond walking to the object.

| Verb | What it does |
|---|---|
| `train` | the learned forecaster on a partition's single runs (`--no-demand`: the demand ablation; `--fraction`: a share of the edit chains) |
| `score` | per world of each partition (default the held-out ones): JSD of each `--net`, the demand model (`--params`), each `--predicted` map, stale and transplanted at other seeds; the across-seed floor; recorded vs predicted change; a PNG per world with `--plots` |
| `curve` | forecast error against post-edit tracks seen, leave-one-seed-out, beside the uniform-prior reference |
| `scaling` | the learned forecaster on 25/50/100 % of the training edits, scored on the held-out partitions |
| `fit` | the demand model's params on a partition |
| `predict` | one world's forecast (`--net`, else the demand model), as cell rows in the map file format or a PNG |

## Placement (INFRA)

[`place.py`](place.py) places the INFRA study's elements (study design §3, §4) through these edits. `CATALOG` holds a charger pair (`charger_1`, `charger_2`: `robot_charger`, 1.0 m apart centre to centre, 175 W, 1.2 m approach), a bay pair (`bay_1`, `bay_2`: `waiting_bay` slots, 1.0 m apart), `locker_1` (`parcel_locker`, 1.5 m access zone, its dock from 0.84 m, beyond the door sweep) and `loading_point` (a `waiting_bay` slot), each object with its jackal dock. The names are the ones the fleet task mode refers to.

`candidate_spots(site, element)` lists, deterministically and cached, a spot every 1 m along every wall piece (`wall_pieces`) of the allowed zones, less the zones NAV@r cuts off at the wheelchair radius (`cut_off_zones`, on walls and doors alone; `cut_off=False` keeps them) and any `exclude` patterns (the twin's INFRA site config names its ramp). A pair is one spot on one piece. A spot carries its objects' poses (fronts into the room, backs at the wall gap), footprints, docks and dock goto points, the clear zone in front and the 1.5 m turning circle beyond the docks, the corridor width it leaves (the free run in front of it: rays along the normal every 0.25 m across its width, each to the first obstacle on the floor raster or out of its zone, the run more than half of the rays find, so a pillar beside it or before one edge does not fail it), and a verdict per §3 rule: `pass` or `fail` where the site has the data, `unknown` (filtering nothing) for outlets, tactile paths, exit width, dead ends, and doors or lifts on a level without any. An object whose model resolves without a bounding box (a floor marking, the twin's `link_scan`) is no obstacle; its dock is. An element model must resolve and carry a `face`.

```python
spots = {name: candidate_spots(site, name) for name in CATALOG}
layout = Layout.of([charger_spot, bay_spot, locker_spot, loading_spot])  # layout.layout_id: a short hash of its spots
check_layout(site, layout)          # failing rules, overlapping elements, a corridor the others narrow
edited = apply_layout(site, layout) # insert (dock, charge power) or move each object, in memory
path = save_layout(site, layout)    # <site key>__layout_<layout id> (a name must keep that prefix), assets hardlinked
placed_elements(layout)             # PlacedElement(name, footprint, dock_point, blocks_floor) for the NAV@r filter
```

[`search.py`](search.py) searches the layouts (§4, §5). `Search(site, loading=(x, y))` pins the loading point (to the nearest valid loading spot) and keeps the valid spots of the other elements, less those over a `keep_out` polygon (PEDS sources and sinks) or clashing with the loading point. Its candidates are the triples of pairwise compatible spots: `check_layout` reads only a corridor window around a spot, so only near pairs are checked. A pick gets the full check, `check_layout` plus `navr_check` against a `NavrBase` of the site. Robot travel runs on the jackal's NAV@r floor at r 0.267 m (`RobotPaths`) and is weighed per job by the §8 job cycle (`TaskMix`).

```python
space = Search(site, loading=TWIN_LOADING_XY, keep_out=sources_and_sinks)
robot_only(space)                   # the valid layout with the shortest expected robot travel per job
joint(space, occupancy, n=2000)     # n random valid layouts by pedestrians under their elements + NAV@r floor loss, top 10
random_valid(space, 5, seed=0)      # the random control
occupancy = Occupancy.from_flow_cells(benchmark_dir, "smu_link_full", episode_s=1500)  # from the site-as-is runs
```

[`infra.py`](infra.py) writes the INFRA Stage 1 suites (the Exposé's mini H1), seed-paired by `flowedit suite`'s stage names `<world>__s<seed>`, both launched with `human.robots.visible: true` and `jackal[4]`:

```bash
flowedit infra reference                               # suite infra_h1_site: the site as is without robots, seeds 1-5; run it first
flowedit infra layouts --occupancy <benchmark dir>      # search, save 7 layout copies (with study_mix), write suite infra_h1
flowedit infra layouts --dry-run                       # the robot-only and random picks without the occupancy, nothing written
```

`layouts` takes the robot-only pick, the best of the joint top 10 and 5 random valid layouts. It saves each as `<site>__layout_<id>` (assets hardlinked; it refuses a local root on another mount) with the scenario copied in. Each copy gets a stage per seed with the robots in `fleet` mode (homes on the bays and chargers, `loading_point`, `locker_1`) and `references: true`, which adds the layout without robots (`unhindered_peds`). It writes `<suite>.layouts.json` beside the suite and prints the benchmark call, whose contest sets the 22 Wh battery. `--rate` defaults to 75 % of the fleet's capacity in the slowest layout at `--speed` (0.5 m/s, the jackal's planner limit).

## Files

| File | Purpose |
|---|---|
| `build.py` | world + scenario to the graph |
| `interactions.py` | how an object is used, from agent type steps and HumanSim's interaction registry |
| `read.py` | graph + settings + seed to a scenario |
| `settings.py` | crowd settings and their defaults |
| `edit.py` | edit operations, sites, saving edited worlds |
| `generate.py` | sampling edits from a seed, geometry-only controls, the manifest |
| `place.py` | INFRA elements, their candidate spots and §3 verdicts, layouts |
| `search.py` | INFRA layout searches: robot-only, joint (occupancy fallback), random valid |
| `infra.py` | the INFRA Stage 1 suites: the site as is, and the searched layouts with the fleet |
| `run.py` | one edit, settings and seed to a recorded run and its tracks |
| `split.py` | split files and the sweep over them |
| `suite.py` | a split as a seed-paired benchmark suite |
| `splits/` | committed splits, and their test crowds' settings in `splits/crowds/` |
| `forecast/` | the flow forecasters: `maps.py` (grid, track binning, JSD), `scene.py` (a site as the forecasters read it), `raster.py` (graph and geometry on the grid), `net.py` (GNN and U-Net), `learn.py` (training, forecasts), `model.py` (the demand model), `evaluate.py` (its fit, recorded maps, transplanting), `validity.py` (scores, baselines, floor, change, tracks seen), `plot.py`, `cli.py` (`flowedit forecast`) |
| `cli.py` | `flowedit build`, `flowedit edit`, `flowedit generate`, `flowedit read`, `flowedit run`, `flowedit sweep`, `flowedit suite`, `flowedit forecast`, `flowedit infra` |
