# AGN RM Project — Student Worksheet

**How to use:** Work top to bottom. Keep all files in your target folder.

**Folder layout (make this first):**
```
<your_target>/
  target_info.md      <- literature + season definitions
  data/               <- survey light curves (wiro + others)
  results/season_x/   <- one folder per season
  figures/            <- final plots & tables for poster and publication
```

"Per season" = repeat the box for each season you defined.

---

### Task 1 — Collect survey light curves
- **Goal:** Get photometry from ZTF, AVA.
- **Do:** Download each; save raw files.
- **Deliver:** files in `data/`.
- **Done when:**  `[ ]` ZTF  `[ ]` AVA (if exists)

### Task 2 — Define seasons
- **Goal:** Split light curves into observing seasons.
- **Do:** Pick JD ranges. Record them.
- **Deliver:** "Seasons" section in `target_info.md` (JD start–end per season).
- **Done when:** `[ ]` every season has a clear JD range  `[ ]` make a plot showing the season cut `[ ]` you will label all later results by season

### Task 3 — Get data ready
- **Do:** Apply reddening correction and rest-wavelength to combined and cali files, keeps names all the same
- **Deliver:** apply to spec files in `data/wiro/` and save them to `data/wiro_reduced` for spec_lc pipeline

### Task 4 — Measure Hβ flux & 5100Å luminosity (spec_lc pipeline)
- **Goal:** Build emission/continuum light curves per epoch.
- **Do:** Run the spec_lc pipeline -> (a) total, (b) velocity-resolved. Repeat all seasons.
- **Deliver:** LC files in `results/season_x/lc/` including continuum and Hβ; comparison plot vs `flux.lst` per season.
- **Per season:** `[ ]` S1  `[ ]` S2  `[ ]` S3 …

### Task 5 — Inter-calibrate continuum (pycali)
- **Goal:** Merge WIRO continuum with other surveys into one LC.
- **Do:** Run pycali. Repeat per season.
- **Deliver:** inter-calibrated continuum LC + pycali plot per season. Put result file `xxx.txt_cali` in `results/season_x/lc/`
- **Per season:** `[ ]` S1  `[ ]` S2  `[ ]` S3 …

### Note: recommend to get uncertainty from median filter method after task 5. At this step, you should have 3 LCs - wiro only continuum, ztf\AVA+wiro continuum, wiro only Hβ


### Task 6 — Time lags (ICCF / or MICA)
- **Goal:** Lag per season, two LC versions.
- **Do:** Run ICCF and/or MICA on (i) WIRO-only, (ii) WIRO + surveys.
- **Deliver:** CCCD + CCF files and plots, both versions, per season. Put result files in `results/season_x/ccf/` and name differently for wiro-only and wiro+surveys.
- **Done when (per season):** `[ ]` WIRO-only result  `[ ]` combined result
- **Per season:** `[ ]` S1  `[ ]` S2  `[ ]` S3 …

### Task 7 — Line width (FWHM & dispersion)
- **Goal:** Measure Hβ width on mean and rms spectra. 
- **Do:** Build mean + rms spectra, subtract the narrow component from mean spectra (subtract_narrow_mean pipeline), and measure FWHM and σ_line for broad Hβ. Per season.
- **Deliver:** FWHM + dispersion table (mean \& rms) per season. Put code/results in `results/season_x/line_width/`
- **Per season:** `[ ]` S1  `[ ]` S2  `[ ]` S3 …

### Task 8 — Black hole mass
- **Goal:** M_BH per season (use the equation provided).
- **Do:** Combine lag (Task 6) + width (Task 7). Adopt the \<f> value from Wu et. al. 2015
- **Deliver:** M_BH value per season. Put code/results in `results/season_x/mass/`
- **Per season:** `[ ]` S1  `[ ]` S2  `[ ]` S3 …

### These additional steps should be done in parallel:

### Task 9 — Literature research
- **Goal:** Know your target before touching data.
- **Do:** Read papers, note prior lags / masses / notes. Present to your colleague.
- **Deliver:** `target_info.md` filled in.

### Task 10 — Publication figures & tables & Poster draft
- **Deliver:** standard figures + tables in `figures/` + poster draft.
- **Done when:** `[ ]` figures  `[ ]` tables `[ ]` draft ready  `[ ]` reviewed by colleague
