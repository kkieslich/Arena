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
| `move` | `obj:<name>` | `x`, `y`, `yaw`, `level` (default: unchanged), `leave_demand: true` (the demand stays at the old spot) | `at`; with `leave_demand` the demand facts, `affords` among them, go to a new `obj:<name>_demand` at the old spot, turned so people queue at the model's front | the pose, for a world object, its height, roll and pitch kept and its `dock`, in its own frame, moved with it; with `leave_demand` its `humansim:` block keeps only `seats`, the dock stays with the model |
| `insert` | `obj:<new name>` | `model`, `x`, `y`, `z` (0), `yaw` (0), `level`, `dock` (where a robot stands to use it, `[[x, y], ...]` in the object frame, see [AUTHORING.md](../../../../AUTHORING.md#4-add-a-scenario)), and any `function` params | entity, `at`, demand | a new world object with its `humansim:` block and `dock` |
| `remove` | `obj:<name>` | none | the entity and every fact about it | the object with its `dock`, and its `kg.yaml` entity |
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

`save` writes the world to the writable local root (`$ARENA_ASSETS_DIR_LOCAL/worlds`, see [worlds/README.md](../../../../worlds/README.md#how-worldidentifier-resolves-a-name)) as `<base>__<what>__<id>`, e.g. `hospital_1__move_fountain__9f4189dc`. `<what>` says what the edits did: the last edit's type, layout `op`, target name and switched-on flags, after `<n>_edits` for a chain (`describe`, at most 40 characters). `<id>` is a hash of the last edit's record, with ints read as floats and false flags (`leave_demand`, `clear`, `remove`) left out. The record names its parent by the parent's key `<base>__<id>` (`Site.key`), so the hash also names the whole chain and the readable part never enters it. The save copies the source world's `assets/`, and writes `flowgraph.yaml` and `edits.yaml` next to `world.yaml`. The scenarios are not copied.

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

- **Determinism:** one `random.Random` per step, keyed by the current world's key (`<base>__<id>`, which names the chain so far), the scenario, seed, step, type and index. Candidates are sorted before every draw and spots rounded to cm, so the same site, seed and code give the same edits and a shorter chain is a prefix of a longer one. No two chains reach the same world at the same step: a duplicate is drawn again, and that retry may take any object, not only one with a need. Footprints come from asset bounds where they resolve, else a 0.8 m box, so a machine with other assets at hand can draw other spots; the manifest carries `generator_version`.
- **On the floor:** spots are drawn on the floor, so a world object standing higher than 5 cm (on a table, a shelf or a wall) is neither moved nor copied.
- **Spots** (`move`, `insert`): inside an allowed zone (not a stair, elevator, lift or shaft), the footprint clear of the zone's edge, walls, door spans (0.8 m) and, by 0.3 m, every object's footprint. A moved object's own old footprint counts too, so the inert copy of its control never overlaps the original. Every spot stands against a wall of its zone: it is drawn along the parts of the zone's edges a wall runs along (within 0.3 m of the edge's line and 10° of its direction, parts shorter than 0.6 m skipped, so a door gap only cuts its edge and a wall stub or a wall meeting the edge gives none), as close as those gaps allow, with a wall behind the middle and both ends of the object's back, and the model is turned so its annotated front (`face` in its annotation.yaml, read as the runtime reads it; without one the side behind its yaw, where the runtime queues it) points away from that wall into the room; a local move draws only along the walls within reach.

| Type | Target | Params drawn |
|---|---|---|
| `move` | an object on the floor, one with a need (`satisfies`) first | a spot in its own room 0.5-3 m away (`scope: local`, 30 %), else in another room of its level (`scope: room`) |
| `insert` | `obj:gen_<model>_<n>`, a copy of an object on the floor, one with a need first | a spot in any room of its level (`scope: room`), the source's demand (service, needs, dwell) and `dock` copied |
| `remove` | an object, one with a need first | none |
| `function` | an object with a need | `capacity`: a new value in the type's range (`CAPACITY`), objects whose type others share first, none with capacity 0. Or `dwell`: a new mean in the type's range (`DWELL_S`, 3-20 s for dispensers), the authored spread scaled to it, only on the sole object for its need |
| `access` | a door whose rooms keep another door, a gated door, or a schedule with a regime | lock a door; unlock or ungate a gated one; a schedule window |
| `layout` | a door | move it 0.5-2 m along its wall; add a 1.5 m door on a wall between two rooms; remove one whose rooms keep another door |

Only objects with a need count as demand: in PEDS only needs send people to an object. `dwell` reaches PEDS through the reader, which writes it as the `duration` of the step at the object. The reader takes one dwell per need, hence the sole-object rule. `satisfies` is not drawn yet. `FIELDS` lists what is drawn and goes into the manifest. Wall paths are not drawn.

**Controls** (`controls(site, chain)`, `--controls`): a function-carrying move or insert gets geometry-only controls, applied to the chain's base, in which only the footprint changes:

| Control | Edit | Result |
|---|---|---|
| `inert_object` | `insert obj:<name>_inert` (move or insert) | the same model and `dock` without demand at the edit's spot, the original left in place |
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

The reader writes the crowd's planner on every person's agent type, and the pilot runs without entrances, so the planner needs no launch argument.

Entrance visitors follow their visitor type, but HumanSim despawns a released agent as soon as it stands inside its target sink, whatever it is doing, and the reader's sink strip (0.1 to 0.6 m inside the door) lies right in front of its source strip (0.8 to 1.8 m). A visitor whose way to its first stop runs along the door's wall crosses the sink within a second and is gone. Until the sink only takes people who leave, [`splits/peds_pilot.yaml`](splits/peds_pilot.yaml) runs without entrances (`settings: {entrances: {rate: 0}}`), so only the reader's placed people walk.

## Paired runs

`flowedit suite <split> [--partition P] [--crowd C] [--settings S] [--out FILE] [--robot R]` writes a split as an arena_evaluation benchmark suite whose runs pair by seed. It generates the split's edited worlds again with the same manifest check as a sweep, reads the crowd into every world at every seed of its partition (the reference runs included), and writes one stage per world and seed. Each stage runs one episode, its `seed` the runtime's episode seed and the reader's seed both, so an edit stage and its base world's stage at that seed run the same people on the same seed. The suite lands in `<imported assets root>/suites/flowedit_<split name>.yaml`, where `--suite flowedit_<split name>` finds it, so a run of it resumes by name. A test crowd (`--crowd C`) is its own suite, `flowedit_<split name>__<crowd>`, of the partitions it runs on: stage names do not say the crowd. `flowedit suite` prints the call:

```bash
flowedit suite splits/peds_pilot.yaml
arena evaluation benchmark --suite flowedit_peds_pilot --contest '[{name: peds_only}]' --lanes 2
flowedit suite splits/peds_pilot.yaml --crowd test_orca
arena evaluation benchmark --suite flowedit_peds_pilot__test_orca --contest '[{name: peds_only}]' --lanes 2
```

| Stage field | Value |
|---|---|
| `name` | `<world>__s<seed>`; an edit stage pairs with `<base world>__s<seed>` (`suite.pairs`) |
| `map`, `seed`, `episodes` | the world, the seed, 1 |
| `config.scenario.file` | the reader's scenario of that world, settings and seed, `flowedit_<scenario>_<settings digest>_s<seed>` |
| `timeout` | the split's `duration`, sim seconds |
| `robot`, `modes.robots`, `config.stationary` | the runner needs a robot in every stage: `--robot` (default `jackal`) parked at (1000, 1000) by the `stationary` mode, as the runner parks its `unhindered_peds` reference runs |

The suite's `launch` runs flatsim headless under lockstep with `human.robots.visible: false`, so the people never see the parked robot, and `references: false`, since the split names its own reference runs. The contestant needs no planner: `[{name: peds_only}]` brings the robot up with the sim's default adapter. `--dry-run` prints the suite without writing a scenario or the suite, and a suite file `flowedit suite` did not write is kept unless `--force`.

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

## Files

| File | Purpose |
|---|---|
| `build.py` | world + scenario to the graph |
| `interactions.py` | how an object is used, from agent type steps and HumanSim's interaction registry |
| `read.py` | graph + settings + seed to a scenario |
| `settings.py` | crowd settings and their defaults |
| `edit.py` | edit operations, sites, saving edited worlds |
| `generate.py` | sampling edits from a seed, geometry-only controls, the manifest |
| `run.py` | one edit, settings and seed to a recorded run and its tracks |
| `split.py` | split files and the sweep over them |
| `suite.py` | a split as a seed-paired benchmark suite |
| `splits/` | committed splits, and their test crowds' settings in `splits/crowds/` |
| `cli.py` | `flowedit build`, `flowedit edit`, `flowedit generate`, `flowedit read`, `flowedit run`, `flowedit sweep`, `flowedit suite` |
