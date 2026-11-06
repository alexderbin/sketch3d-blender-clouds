# Clouds — procedural volumetric cloud generator for Blender: cumulus to cumulonimbus, composed in the frame

> Part of [sketch3d](https://github.com/alexderbin/sketch3d-blender) — AI-ready 3D automation for Blender: procedural scenes from YAML recipes. This repository is a standalone slice: the `clouds` component with the sketch3d core it runs on. An AI agent can drive it by [AGENTS.md](AGENTS.md).

Clouds is a procedural volumetric cloud generator for Blender, written in Python: it builds clouds by meteorological
type as OpenVDB volumes, places them in a camera frame by a composition recipe and lights them offline.
Clouds are volumes, not a sky texture: each type — fair-weather cumulus to cumulonimbus with an anvil, lenticular,
stratocumulus, altocumulus, cirrus — is a numpy density field shaped the way a painter builds a cloud mass, placed by a
composition recipe written in fractions of the frame, and lit in advance so that Eevee renders it as a cheap
"absorption + emission" volume. Light is solved offline (sun march, single Henyey–Greenstein scattering, multiple
scattering by diffusion — Stam 1995) and checked against a path-traced reference render of a sample cloud.
The `clouds` step reads the cameras — the composition lives in a camera's frame — and adds cloud volumes to the light
section of the scene.

**Use it to:**

- Add volumetric cumulus clouds to a Blender sky render: `clouds: {recipe: spacious}` (one tower, a counterweight, much
  sky) or `clouds: {recipe: rich}` (large layered banks across the frame)
- Get a storm front, a fair-weather sky, an afterglow or a night sky over mountains from a ready composition:
  `clouds: {recipe: castle_storm}`, or `castle_fair`, `castle_afterglow`, `castle_night`
- Compose your own sky with towering cumulus, a cumulonimbus with an anvil or lenticular clouds: a new recipe in
  `components/clouds/recipes.yaml` with roles `{type, U, width, top, dist, seed}`, types from `types.yaml`
- Check a composition for touching masses and equal tops before rendering, without Blender:
  `components.clouds.api.check(cam, roles)` or `.venv/bin/python -m components.clouds.bench.bench`
- Light your own cloud density field offline (sun march, scattering by diffusion) and get an emission grid for an
  "absorption + emission" volume: `components.clouds.api.form` and `components.clouds.api.bake`
- Draw a rainbow or a lightning bolt over a finished frame, without Blender: `components.clouds.rainbow.apply` and
  `components.clouds.lightning.draw`

## Features

- **12 types** (`types.yaml`): humilis, mediocris, bank, bank_calm, bank_far, congestus (a leaning tower), cumulonimbus
  (tower + anvil drifting downwind), lenticular (stacked smooth lenses), frame (a tall mass at the frame edge),
  stratocumulus and altocumulus (cell layers), cirrus (strokes along the wind).
- **Masses as an artist builds them** (`forms.mass_lobes`): large soft domes with 2–5 levels of billows — the number of
  levels follows the size in frame — grouped accents instead of a "string of beads" edge; towers, anvils, lenses.
  More forms: cell layers cut by the terrain, cirrus layers, mammatus, rain shafts, a storm ceiling seen from below,
  storms made of small cells.
- **Composition in the frame** (`compose.py`): a role is a type + centre U, width and top V as fractions of the frame +
  distance; the solver turns it into azimuth, length and height in the world (screen width = length + 2 dome radii) and
  `check` flags touching masses and tops at the same level before Blender runs.
- **Light in advance** (`light.py`): march to the sun with doubling steps (Schneider 2015), single HG scattering, multiple
  scattering by diffusion on a coarse grid (δ-scaling, Marshak boundary, P1 radiance towards the camera, exchange
  through gaps), light-and-shade detail below the cell size; the moon as a second source. Mean error 0.089 against a
  path-traced reference render (`bench/reference_render.py`).
- **Lightning** (`lightning.py`): channel by midpoint displacement with branches, a "spider" along the cloud base and
  in-cloud branches that light the mass from inside; strike point by visibility from the cameras; the bolt drawn per
  frame (core, halo, air glow) with occlusion. Fires and lamps can light clouds from below.
- **Parallel baking** (`bake.py`): cloud forms and their light are independent numpy jobs for a process pool; Blender
  only writes OpenVDB grids and renders.
- **Post tools without Blender**: rainbow from scattering angles (primary bow 40.7–42.3°, Alexander's dark band,
  secondary bow 50.4–53.4°) — `rainbow.py`; six cube faces → an equirectangular 2:1 skybox — `skybox.py`; re-mixing of
  light passes (rim, gaps, glow, flash) without re-baking — `paint.py`.
- **Clouds from the terrain** (`terrain.py`, `anchor.py`): valley and ridge levels, relief range, peaks and mountain
  spacing from the scene's far heights; regimes fair, peaks, belt, fog — clouds tied to landforms, one set for any camera.

## How it works

Plan (Python): the recipe from `recipes.yaml` is solved against camera `camera` of the scene's camera layer → roles in
the world; the scene's light plan (`core.light.far.plan_of`: sun, sky, ground) is written next to it; warnings of the
solver go to the journal. Build (Blender, `bl_clouds.py`): for each mass role — form (`api.form`) → light in advance
(`api.bake`, emission σρ·J in RGB) → an OpenVDB grid with density and emission → a volume `cloud_<i>` with an
"absorption + emission" material. Knowledge: `types.yaml` (form knobs of each type), `recipes.yaml` (compositions).

## Steps and keys

| Step | Reads → writes | Keys (defaults) |
|---|---|---|
| `clouds` | cameras (+ the scene's light plan) → `light`: volumes `cloud_<i>`, VDB in the step folder | `recipe` (spacious), `camera` (0), `density` (0.35 — soft, no banding), `detail` (0.3) |

Recipes for the step: `spacious` (one tower, a counterweight, much sky), `rich` (large layered masses across the frame),
`type_mediocris`, `castle_fair`, `castle_rich`, `castle_storm`, `castle_afterglow`, `castle_night`. Off in the
`sketch` profile. All keys with descriptions: [component.yaml](/components/clouds/component.yaml)

## Usage

```yaml
steps:
  - relief: {preset: river_crossing, seed: 1}
  - camera: {anchor: x.ford, azimuth: 205, elevation: 4, distance: 160, lens: 28, res: 960x540}
  - clouds: {recipe: rich, camera: 0}
  - light: {}
```

```sh
.venv/bin/python -m components.clouds.bench.bench      # no Blender: diffusion energy, recipe solver, type forms
blender -b --factory-startup --python-exit-code 1 -P core/blender/run.py -- components/clouds/bench/stand.py -- out/dev/clouds/rich
.venv/bin/python -m components.clouds.bench.ceiling_persp <out.jpg> '<json list of variants>'
```

The stand takes the recipe, a silhouette (notan) check, time of day, weather, lightning and the canyon and vortex sky
compositions as key=value arguments listed in its docstring; `bench/reference_render.py` compares the light with the
reference render; `bench/cells_light.py` and `bench/ceiling_persp.py` are fast numpy stands for storm cells and storm
ceilings.

## Driving it from an AI agent

`clouds` goes after a `camera` step and takes its light from the scene's `far.yaml`, not from a later `light` step:

```yaml
  - camera: {anchor: x.ford, azimuth: 205, elevation: 4, distance: 160, lens: 28, res: 960x540}
  - clouds: {recipe: castle_storm, camera: 0, density: 0.35, detail: 0.3}   # names: components/clouds/recipes.yaml
```

- Check without Blender: `.venv/bin/python -m core.manager.pipeline <recipe.yaml> --dry`. All keys, defaults and
  values: `components/clouds/component.yaml`.
- A run prints `__RESULT__{"pipeline", "out", "failures", "done", "cached", …}` as its last line; the run folder holds
  `journal.md` and, with the `frames` output, the frames and a contact sheet `contact.jpg`.
- Request to keys: "a storm front over the ford at dusk" → `recipe: castle_storm` plus
  `far: {time: dusk, weather: storm}` in the scene's `far.yaml`; "a calm summer sky" → `recipe: castle_fair`; softer or
  denser clouds → `density`.

## Files

| File | Role |
|---|---|
| `forms.py`, `types.yaml` | density fields of every type and form |
| `compose.py`, `recipes.yaml` | composition in the frame and its recipes |
| `light.py`, `bake.py` | light in advance; process-pool baking |
| `lightning.py`, `rainbow.py`, `skybox.py`, `paint.py` | lightning; rainbow, skybox and paint post tools |
| `terrain.py`, `anchor.py` | clouds tied to the terrain |
| `bl_clouds.py`, `steps.py`, `api.py` | Blender volumes; step entry; public entry without Blender |

## Assets

The `clouds` step is fully procedural. The stands, the reference bench and the hybrid storm recipes use a sample cloud
volume in `library/clouds/refvol/` (`refvol_cloud_eighth.vdb`, the reference scene `.xml` and its render `.exr`), not
included.

## Limitations

- The step builds mass types only; layer types (stratocumulus, altocumulus, cirrus) are forms used by the stands.
- Clouds are lit by the scene's own light description (`far.yaml`, else the site biome); `time` and `weather` keys of a
  later `light` step do not re-bake them, and the light node may add its own sprite clouds for its weather.
- The terrain-anchored scene driver that reads the `castle_t_*` recipes is not part of this repository; `lightning.py`,
  `paint.py`, `rainbow.py` and `skybox.py` expect run folders written by such a driver.
- No cloud shadows on the ground and no animation (drift, growth) yet; volumes cost 5–30 s of light per frame and
  0.3–1.5 GB of VDB per scene.

## FAQ

**How do I make volumetric clouds in Blender with Python?** Add a `camera` step, then `clouds: {recipe: rich}`. The step
shapes a density field per cloud, bakes its light and writes OpenVDB volumes `cloud_<i>` with an "absorption + emission"
material that Eevee renders.

**How do I get a cumulonimbus with an anvil, a lenticular cloud or a towering cumulus?** Name the type in a role of a
recipe in `components/clouds/recipes.yaml` (`type: cumulonimbus`, `lenticular` or `congestus`); the shape knobs are in
`components/clouds/types.yaml`. The step builds mass types only: stratocumulus, altocumulus and cirrus layers are forms
used by the stands (`components/clouds/bench/stand.py`).

**Can I use it without the rest of sketch3d?** Yes, it also ships as its own repository, `sketch3d-blender-clouds`, with the
core it needs. Shapes, the composition solver and the baked light are plain Python and numpy
(`components/clouds/api.py`: `layout`, `check`, `form`, `bake`); the `clouds` step reads a scene's `cameras.json`,
which the `camera` step writes.

**Does it need a GPU?** Shapes and light are computed on the CPU with numpy (0.1–0.5 s per mass for the shape, 5–30 s
per frame for the light); Eevee renders the volumes and is faster with a GPU.

**How do I check a sky layout before rendering?** Run `.venv/bin/python -m components.clouds.bench.bench`: it solves the
recipes that carry their own camera (`spacious`, `rich`, `type_mediocris`), reports touching masses and tops at the same
height, and checks the diffusion energy and the shape of every type. For the scene camera the step puts the same
warnings into `clouds_plan.json`.

## Install

Blender 5.2 as `blender` on `PATH`, Python 3.13, `ffmpeg`.

```sh
git clone https://github.com/alexderbin/sketch3d-blender-clouds.git && cd sketch3d-blender-clouds
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m core.services.venv_root
.venv/bin/python -m core.light.far agx      # bake the AgX look tables once (runs Blender for ~10 s)
.venv/bin/python -m pytest tests
```

Commands above run from the repository root. How recipes, steps and layers work — [ARCHITECTURE.md](ARCHITECTURE.md);
how an AI agent runs, checks and reads the pipeline — [AGENTS.md](AGENTS.md).

## License

MIT — see [LICENSE](LICENSE).
