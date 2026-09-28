# Probe verification notes (M10.2-era)

Distilled from the M10.2 "invisible vegetation" debugging session — keep these
when running future screen probes.

## Verified observation points
- LAND (forest biome, terrainHeight ~402 m): `lat=-38 / lon=120`
  (found by evaluating the sim's own terrainHeight/fbm3 in node — see
  `.tmp_findland.mjs`, kept local via .gitignore)
- SEA — do NOT use for vegetation/ground-color tests: `lat=48/lon=112`,
  `lat=26/lon=114`. A sea point mimics "nothing renders".

## Probe kinds (testAuto.ts `?probe=`)
- `m` — 12x8 pixel color matrix (ground appearance)
- `d` — center rgb + planet hit count (planet=3/3 expected)
- `g` — geometry anomalies out of 36 (anomalies=0/36 expected)
- `j` — frame stability: changed/1035 px, rebases (changed=0, rebases=1 expected
  when the scene is settled)

## Regression set
- Moon landing: `?demo=lunar&moonangle=0&speedup=4` -> HUD `TOUCHDOWN LANDED`,
  GS 0.0, AGL 0 m
- Orbit: `?demo=orbital&moonangle=0` -> HUD elements match patched-conic theory

## TLI numeric mission check (historical, script was temporary)
Parking orbit a 6571 km, e 0.000000, T 88.35 min, 1-period drift 0.000 m.
TLI dv 3133 m/s, prop left 1644 kg (Tsiolkovsky-exact). Transfer ra 384.4 Mm,
T 238.9 h. SOI entry t+64.19 h at 66.16 Mm, radial -583.9 m/s. Capture dv
595 m/s, prop left 1057 kg. Lunar orbit rp 500 km altitude, e 0.9346; 3 orbits
no impact.
Known open item: lunar lander demo lands with PROP 1859 kg vs the numeric
mission's 1057 kg — the demo scenario starts from different initial conditions;
worth reconciling when the full TLI->landing scenario is scripted end-to-end.
