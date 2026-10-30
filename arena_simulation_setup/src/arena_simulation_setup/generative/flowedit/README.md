# FLOWEDIT graph

One graph per site with each object's demand on its node, what each object is for, and each door's access. It is a [KG](../../../../AUTHORING.md#kg) scene (`arena-kg/1`) over `core@2`, `space@1` and [`flow@1`](../../kg/packages/flow@1.toml), kept in its own `flowgraph.yaml`, never in the world's `kg.yaml`. Edits change it, a forecaster reads it, simulators never read it. Persons stay in agent types.

```
world.yaml (geometry)          --+
scenario objects (humansim:)   --+-- build --> flowgraph.yaml
kg.yaml (room purposes, read)  --+
```

## CLI

`flowedit build <world> [<scenario>]`, or `python3 -m arena_simulation_setup.generative.flowedit.cli build ...`. A dry run prints the graph, `--write` writes it to `flowgraph.yaml` next to the scenario's `scenario.yaml` (or `--out`). Worlds resolve as in [generative/scenario](../scenario/README.md#cli).

```bash
flowedit build hospital_1 queue --write
```

`flowedit edit <world> <type> <target> [-p KEY=VALUE ...] [--scenario <scenario>]` applies one [edit](#edits). A dry run prints the edit record and the world name, `--write` saves the world. `-p` reads VALUE as YAML and nests dotted keys (`-p satisfies.thirst=80`). An edited world is edited again by its name; its scenario comes from its `edits.yaml`.

```bash
flowedit edit hospital_1 move obj:fountain --scenario queue -p x=15 -p y=11.5 --write
flowedit edit hospital_1__a9f8ee8a layout door:or_door -p op=remove --write
```

## Graph

| Node | Holds | From |
|---|---|---|
| `room:<zone>` | `has_purpose purpose:<role>` | the world KG's `has_purpose`, else the substring rules of [`zones.py`](../scenario/zones.py) as `params: {by: rule}` |
| `door:<name>` | `connects` between two rooms | derived from geometry |
| `obj:<name>` | prop `model`, `at floor:<level> {x, y, yaw, pose_ref}` | `world.yaml` objects and the scenario's `objects:` |
| `obj` demand | `serves service:<type> {capacity, interaction_radius}`, `satisfies need:<need> {amount}`, `has_formation formation:<type> {<params>, anchor_kind, anchor_ref}` | the object's `humansim:` block |
| `obj` dwell | `dwells <mean s> {std, clip_low, clip_high, from}` | the `duration` of the first agent type step whose `target` is the object's name, else its type |
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

## Edits

[`edit.py`](edit.py) has one operation per edit type. Each changes the graph and the world's geometry together, then recomputes `contains` and `connects`. A **site** is a world without its scenario objects, plus its graph (`open_site`). `apply(site, type, target, params)` returns a new site and leaves the old one unchanged. `save(site)` writes it.

| Type | Target | Params | Graph | `world.yaml` |
|---|---|---|---|---|
| `move` | `obj:<name>` | `x`, `y`, `yaw`, `level` (default: unchanged) | `at` | the pose, for a world object |
| `insert` | `obj:<new name>` | `model`, `x`, `y`, `yaw` (0), `level`, and any `function` params | entity, `at`, demand | a new world object with its `humansim:` block |
| `remove` | `obj:<name>` | none | the entity and every fact about it | the object, and its `kg.yaml` entity |
| `function` | `obj:<name>` | `type`, `capacity`, `interaction_radius`, `satisfies: {<need>: amount or null}`, `dwell: seconds, {mean, std, clip_low, clip_high} or null` | `serves`, `satisfies`, `dwells` (`from: edit:<id>`) | the `humansim:` block, for a world object |
| `access` | `door:<name>` | `locked`, `authorized`, `unlock_on`, or `remove: true` | access facts | the door's `gate` preset (new gates start unlocked) |
| `access` | `schedule:<name>` | `regime`, `windows`, `default`, or `remove: true` | access facts | the `schedules:` entry |
| `layout` | `door:<name>` | `op: add` (`level`, `span`, door fields), `move` (default: `span` or `offset: [dx, dy]`, door fields), `remove` | the door entity | the door |
| `layout` | `wall:<level>/<index>`, `wall:<level>` to add | `op: add` (`path`, `closed`, `kind`, `material`), `move` (`path` or `offset`), `remove` | none | the `walls:` path at that index |

- **Scenario objects** (the fountain of `queue`) change only in the graph. A reader from graph to scenario (not built yet) is to write them into the scenario.
- **Doors:** a moved or added door must lie on a wall. A door moved or removed from a gap left in its wall gets the gap closed with a wall styled like the wall at its start. Walls drawn through a door are cut at runtime anyway.
- **Walls:** only `walls:` paths count. Walls from a zone's `walls: outline` are not addressable.
- **Checks** stay basic: unknown target, taken name, door off a wall, then `World.validate` and the graph's vocabulary. Rules about rooms or overlap belong to the edit generator.

`save` writes the world to the writable local root (`$ARENA_ASSETS_DIR_LOCAL/worlds`, see [worlds/README.md](../../../../worlds/README.md#how-worldidentifier-resolves-a-name)) as `<base>__<id>`. `<id>` is a hash of the last edit's record, which names its parent world, so it also names the whole chain. The save copies the source world's `assets/`, and writes `flowgraph.yaml` and `edits.yaml` next to `world.yaml`. The scenarios are not copied.

```yaml
format: flowedit-edits/1
base: hospital_1
scenario: queue
seed: 0
edits:
- {id: a9f8ee8a, parent: hospital_1, type: move, target: 'obj:fountain', params: {x: 15, y: 11.5}}
- {id: d1fe1a64, parent: hospital_1__a9f8ee8a, type: layout, target: 'door:or_door', params: {op: remove}}
```

## Files

| File | Purpose |
|---|---|
| `build.py` | world + scenario to the graph |
| `edit.py` | edit operations, sites, saving edited worlds |
| `cli.py` | `flowedit build`, `flowedit edit` |
