# FLOWEDIT graph

One graph per site with each object's demand on its node: what each object is for. It is a [KG](../../../../AUTHORING.md#kg) scene (`arena-kg/1`) over `core@2`, `space@1` and [`flow@1`](../../kg/packages/flow@1.toml), kept in its own `flowgraph.yaml`, never in the world's `kg.yaml`. Edits change it, a forecaster reads it, simulators never read it. Persons stay in agent types.

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

## Graph

| Node | Holds | From |
|---|---|---|
| `room:<zone>` | `has_purpose purpose:<role>` | the world KG's `has_purpose`, else the substring rules of [`zones.py`](../scenario/zones.py) as `params: {by: rule}` |
| `door:<name>` | `connects` between two rooms | derived from geometry |
| `obj:<name>` | prop `model`, `at floor:<level> {x, y, yaw, pose_ref}` | `world.yaml` objects and the scenario's `objects:` |
| `obj` demand | `serves service:<type> {capacity, interaction_radius}`, `satisfies need:<need> {amount}`, `has_formation formation:<type> {<params>, anchor_kind, anchor_ref}` | the object's `humansim:` block |
| `obj` dwell | `dwells <mean s> {std, clip_low, clip_high, from}` | the `duration` of the first agent type step whose `target` is the object's name, else its type |

`contains` and `connects` are derived as when a world's KG loads. Unlike `kg.yaml`, the graph carries each object's `model` and pose, a copy of `world.yaml` that `flowedit build` refreshes; rooms and doors carry no geometry. Poses are in the level frame. A scenario object posed at a zone name gets the point the runtime would sample under `--seed`, and keeps the name as `pose_ref`. Dwell is read from the scenario-local agent type files (`./<type>.yaml`) its humans and regions name, and a note names other steps that give the same object another duration. `seats` and formations anchored at a pose are not in `flow@1`.

```yaml
- {subject: 'obj:fountain', verb: at, object: 'floor:0', params: {x: 11.0, y: 10.0, yaw: 1.57}}
- {subject: 'obj:fountain', verb: serves, object: 'service:fountain', params: {capacity: 1}}
- {subject: 'obj:fountain', verb: satisfies, object: 'need:thirst', params: {amount: 100.0}}
- {subject: 'obj:fountain', verb: has_formation, object: 'formation:line', params: {base_step: 1.0, front_offset: 0.8}}
- {subject: 'obj:fountain', verb: dwells, object: 5.0, params: {std: 1.5, clip_low: 4.0, clip_high: 6.0, from: 'colleague.yaml#drink.queue_and_drink'}}
```

Load and save it with `arena_simulation_setup.kg.load` and `write`.

## Files

| File | Purpose |
|---|---|
| `build.py` | world + scenario to the graph |
| `cli.py` | `flowedit build` |
