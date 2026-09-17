# What changed in this pass

## Notebook (`python/student_project.ipynb`)
- **Fixed a real bug**: the "ONI correlation and regression" section had a docstring copy-pasted from the previous (regional analysis) section and duplicated ~90 lines of file-reading/anomaly code that had already run one cell earlier.
- **Refactored so the data loads once.** A new "0b. Load the data" cell reads every NetCDF file a single time and every later section reuses the cached arrays/DataFrames. Previously sections 2, 3, 4 and 5 each re-read and re-processed the same 21 years of files from scratch — four full passes over the data where one is needed. This should noticeably speed up the notebook on Binder.
- **Unified the region definitions.** The maps (section 2) and the regional time-series analysis (section 4) used two different, slightly inconsistent sets of region boundaries under different names (e.g. "Niño 3.4" vs "Nino3.4", with different lat/lon boxes for the stratocumulus regions). There is now one `REGIONS` dictionary of 12 regions, numbered consistently, used everywhere.
- **Fixed a labelling problem on the maps**: with 11+ region boxes drawn on a single global map, full text labels overlapped illegibly in the crowded tropics. Boxes are now numbered (1-12) with a small legend table (`REGIONS_TABLE`) printed once in Setup and again after the maps, so the numbers are easy to look up.
- Figures now save consistently to `figures/` and result tables to `results/` (previously some scripts saved to the working directory directly, others to `plots/`/`results/`).
- Verified the whole notebook executes top-to-bottom without errors, using synthetic ERA5-like data built to the same file/variable conventions (not included here — this was only for testing).

## Slides (`guidance/*.pptx`)
- Fixed two schematic images that had visible rendering defects: `report_funnel_better.png` had text overflowing its box border, and `walker_course_schematic.png` had labels overlapping the arrows/cloud symbol. Both are used repeatedly across the slides and the handbook, so this one fix propagates everywhere.
- Filled in the one slide that had no visual (`01_Project_launch.pptx`, "Today: get something working") using the existing-but-unused `project_workflow.png` asset.
- Replaced the three "SCIENCE IMAGE" placeholder boxes' source notes with specific, direct links to real NOAA imagery (see below) so you can go straight to the source rather than searching.

## Handbook (`guidance/ENSO_cloud_project_student_guide.docx`)
- Same two schematic image fixes applied (it embeds its own copies of the same assets).
- Content/structure was already strong — no changes needed there.

## Curated image links (for the three "SCIENCE IMAGE" slides)
1. **El Niño / La Niña SST anomaly comparison** (`01_Project_launch.pptx`, slide 4):
   https://www.climate.gov/news-features/understanding-climate/climate-variability-oceanic-nino-index
   Direct image: the Dec 1997 (El Niño) vs Dec 1988 (La Niña) SST maps partway down the page.
   Credit: "Maps by NOAA Climate.gov, based on data from NOAA's Physical Science Lab."

2. **Niño 3.4 region map** (`03_ENSO_analysis.pptx`, slide 2):
   Same page as above: https://www.climate.gov/news-features/understanding-climate/climate-variability-oceanic-nino-index
   Direct image: the Niño 3.4 box map near the top of the page.
   Credit: "NOAA Climate.gov image by Fiona Martin."

3. **Marine stratocumulus off western South America** (`04_Choose_your_question.pptx`, slide 3):
   https://www.nesdis.noaa.gov/our-environment/clouds/marine-stratocumulus-clouds
   This is the official NOAA/NESDIS article with the NOAA-20 image from 3 Dec 2019 that the slide's own caption already anticipated.

All three are official NOAA pages, safe to hotlink/cite and free of usage restrictions for teaching.

## Not changed
- The wording/structure of the six weekly decks and the handbook's sections were already well-judged for the level and the CORE/EXPLORE/OPTIONAL framing was kept exactly as designed.
- `Canvas_weekly_plan.md` and the handbook's own "Tutorial roadmap" table already map cleanly onto your eight timetabled dates: 29 Sept, 6 Oct, 13 Oct, 27 Oct and 10 Nov, 17 Nov each pair with one of the six numbered decks (01-06); 24 Nov and 1 Dec are deliberately no-new-content drafting/final clinics, so all eight dates are accounted for without needing two more decks.

## Second pass: fitting every session into its 50-minute slot
- **Added timing to every deck's speaker notes** (Presenter View / Notes — invisible on
  the projected slide). Each deck now totals 50 minutes: content minutes per slide + one
  hands-on block + a 5-minute buffer. Deck 06 (Writing workshop) carries the most
  front-loaded teaching (~16 min across 6 slides) and is flagged as the one to watch if
  running behind.
- **Added a "Timing (50 min)" line to every week in `Canvas_weekly_plan.md`**, plus a
  triage suggestion for the two clinic weeks (24 Nov, 1 Dec) that have no slides to pace
  against (roughly `45 min / class size` per student, said out loud at the start).
- **Added the same summary as a short note in the handbook**, right after the "Tutorial
  roadmap" table.
- **Found and fixed a genuine timing/reliability risk**: tested the refactored notebook
  end-to-end and hit it directly — the maps section (13 October) calls Cartopy, which
  downloads map-outline data from a third-party host the first time it's used in a
  session, and that download failed outright in the test environment. Added a
  `postBuild` file at the repo root (next to `environment.yml`) that pre-fetches this data
  when Binder builds the image, so students never depend on a live download mid-class.
  This needs to be at the **repository root** (same level as `environment.yml`), not
  inside `analyses-project/`, for Binder to pick it up — it's included there in this
  package.
- **Added a tutor tip** to `analyses-project/python/README.md`: launch the Binder link
  yourself 15-30 minutes before each session (especially 29 September, everyone's first
  launch) so students' launches reuse the cached build instead of triggering a slow
  rebuild.
- One line added to the notebook's maps-section markdown, telling students the map
  outlines should draw instantly and to flag their tutor if one instead pauses on
  "Downloading..." (a sign the build-time cache wasn't picked up).

## Third pass: fixing a genuine rendering bug and a full visual design pass

This pass responds directly to two pieces of feedback: the decks and handbook weren't
good enough, and the earlier "curated image links" pointed at article/landing pages
rather than links you could actually click "download image" on.

- **Found and fixed a real authoring bug behind the ugly/clipped text you saw.**
  Almost every text box in all six decks (102 of 111 text frames) had word-wrap
  switched off and "shrink box to fit text" switched on at the same time -- two
  settings that fight each other, and the result was text rendered as a single
  unwrapped line that overflowed past the edge of the slide. This wasn't a PowerPoint
  vs. LibreOffice quirk, it was baked into the file. Fixed by turning word-wrap back on
  and fixing the box sizes, in every text box, in every deck.
- **Replaced the three landing-page links with direct image-file links.** The
  "SCIENCE IMAGE" placeholder boxes (El Nino/La Nina SST comparison, Nino 3.4 map,
  marine stratocumulus off Peru) now link straight to the actual image file, not the
  article it's embedded in, so "right-click -> Save Image" / "download" works from the
  slide itself:
  - https://www.climate.gov/sites/default/files/2025-06/ClimateDashboard-variability-Oceanic-Nino-Index-image-20210505-FB.jpg
  - https://www.climate.gov/sites/default/files/styles/full_width_620_alternate_image/public/Fig3_ENSOindices_SST_610.png
  - https://assets.science.nasa.gov/content/dam/science/esd/eo/images/imagerecords/83000/83796/Peru_tmo_2014148_lrg.jpg
  Each placeholder box was also restyled (dashed border, tinted fill, clear "Download
  this image and insert it here:" instruction + credit line) so it reads as a deliberate
  placeholder rather than an unfinished slide.
- **Real visual design, not just bug fixes.** Every deck now has:
  - its own accent colour (Tutorial 1 deep teal, 2 blue, 3 burnt orange, 4 green, 5
    purple, 6 amber) so a slide is identifiable as "week N" at a glance;
  - a coloured title-slide cover with white typography and a "Tutorial N of 6" tag;
  - a slim identification banner across the top of every slide (course name, tutorial
    number, slide count);
  - a coloured title + rule on every content slide, replacing plain unstyled black text.
  - Removed a leftover authoring mistake where a generic "Course-team schematic /
    teaching material" caption had been copy-pasted onto every slide, including slides
    that have no image on them at all.
- **The handbook got the matching treatment**: a coloured cover banner and heading
  palette that matches the decks (teal, not the Office default purple), the eight
  timetabled tutorial dates printed on the cover, a page footer with page numbers, a
  Table of Contents (right-click -> Update Field in Word to populate/refresh it), and a
  shaded header row on the tutorial roadmap table.

Net effect: same content and structure as before (which you'd already said was
well-judged), but the files now look like they were designed, the placeholder images
have real, directly downloadable links with credits, and the text-clipping bug -- which
was a genuine defect, not a rendering artefact -- is gone.

## Fourth pass: real photos embedded, more schematics, notebook screenshots

You sent through the three photos downloaded from the links above, plus asked for more
schematics and some screenshots of the notebook to help students get started.

- **The three real photos are now embedded directly in the slides** (not just linked):
  El Nino/La Nina SST comparison (deck 01), the Nino 3.4 region map (deck 03), and the
  marine stratocumulus image off Peru (deck 04) each replace their old placeholder box,
  bordered in the deck's accent colour with the credit line underneath. The Peru photo
  came in at 4400x5600 (5.2MB) -- resized to 1100x1400 (~490KB) before embedding so it
  doesn't blow out the file size; visually identical at slide scale.
- **Eight new schematics added** to slides that previously had no figure at all:
  - "What a good scientific figure shows" -- an annotated mock figure (title, axis
    labels, colour scale, caption) matching the four bullets on that slide exactly.
  - "Write around evidence" -- a Claim -> Evidence -> Interpretation -> Qualification
    flow diagram, same visual style as the existing report-structure diagram.
  - Five "wrap-up" cards (Tutorials 2-6's closing goal/milestone/check slides) -- a
    consistent checklist or fill-in-the-blank card in the deck's own accent colour,
    reusing the slide's own bullet text so nothing was invented.
  - Deck 06's "funnel towards the question" slide now reuses the existing
    report-funnel schematic, since that's literally what it's describing.
  - Fixed a layout bug this introduced: a few of these slides had a full-width body
    text box left over from when they had no image (so long bullets ran on under
    where the new picture now sits). Narrowed those text boxes to the same
    two-column width every other slide already uses.
- **Real notebook screenshots**, captured by actually running the notebook (on
  synthetic test data, purely to get genuine output) and screenshotting the rendered
  result: the opening brief, the "0b. Load the data" cell succeeding, the global ENSO
  maps, and the "STOP: choose your investigation" cell. These now appear in deck 01
  ("Today: get something working") and as a new "What you'll see when you open it"
  section in the handbook, right after "What you will learn".
- **Found and fixed a real bug while capturing those screenshots**: the maps-section
  code cell printed a literal `\nReminder -- ...` (backslash-n as visible text) instead
  of a line break, because of a doubled escape character in the source. Every student
  running the notebook would have seen that typo. Fixed in `make_notebook.py` and
  regenerated `student_project.ipynb`.

All files are still small: the largest deck is 720KB (the one with two embedded
photos), the handbook 1.0MB, full package still a few MB.

## Fifth pass: clearer schematics, real satellite imagery, weekly orientation, and a marking scheme

You asked for more/better schematics (especially the latitude area-correction one),
questioned whether the repeated "Sources and attribution" slide was needed, wanted
real imagery of deep convection/anvils and a global cloud-cover view, and asked for
each week's purpose to be really clear plus a proper marking scheme with room to
pass easily at 40% but real headroom above a "solid ~65%" report.

- **Replaced the weak `cosine_weighting.png`.** The old version was just an empty
  flat grid asserting the point in a caption. The new one actually shows it: a flat
  array grid next to a real cross-section of the sphere, with the same two latitudes
  marked on both, so you can see the 0.5x shrinkage at 60N rather than take it on
  faith. Re-embedded into deck 02's "Why area weighting matters" slide.
- **Built a new, concrete deep-convection schematic** (`deep_convection_schematic.png`)
  showing the physical picture -- warm pool SST, a rising cumulonimbus tower, the
  anvil spreading at the tropopause (~16 km), versus the subsiding side with only
  thin low cloud capped at the boundary layer (~2 km). This complements (doesn't
  replace) the existing abstract Walker-circulation boxes and cloud-level ellipses.
  Added as its own new slide in deck 04, right after "Different cloud levels,
  different processes".
- **Found and verified three real, direct-download NASA image links** (not landing
  pages) for a global full-disk cloud-cover view and two real anvil/deep-convection
  photographs, and added them as clearly-marked "download and insert" placeholder
  slides -- the same pattern used successfully for the three photos you already sent
  back:
  1. Global cloud cover ("Blue Marble" composite) -- deck 01, new "The bigger
     picture" slide.
  2. Cumulonimbus anvil over West Africa (NASA ISS016-E-27426) -- deck 04, new
     "Anvils, seen from space" slide.
  3. Thunderstorms over Borneo/Maritime Continent (NASA ISS040-E-88891) -- same
     slide, alongside #2.
- **Removed the redundant "Sources and attribution" slide from decks 02-06.** It was
  byte-for-byte identical across all six decks -- confirmed by direct inspection, not
  assumption. It now appears once, in full, at the end of deck 01, with a one-line
  pointer added to the speaker notes of each other deck's new last slide.
- **Added a "Today's plan" slide to all six decks**, right after the title slide:
  the week's one-line goal, a numbered "what you'll do today" list, and a small
  6-dot progress indicator with the current week highlighted in that week's own
  accent colour -- so it's immediately clear what each session is for and what to
  actually do.
- **Added a real marking-criteria rubric**, using UK percentage bands: 40-49% Pass,
  50-59% (2:2), 60-69% (2:1 -- deliberately calibrated so a report that does
  everything correctly sits here, at a solid ~65%), 70-79% (First), 80%+
  (Exceptional). Full descriptors are new handbook section 9, "How your report will
  be marked" (existing sections 9-11 renumbered to 10-12 accordingly). A condensed
  version appears as its own slide in deck 01 ("How your report will be marked", so
  students see the scheme on day one) and again as a short recap in deck 06, right
  before the final "Conclusion and final check" slide.
- **Renumbered every "Slide i of N" footer** across all six decks to match the new
  slide counts after these insertions/removals.
- Net slide-count changes: deck 01 seven to ten slides, deck 04 seven to nine, deck
  06 eight to nine; decks 02/03/05 unchanged in count (one slide added, one removed).

**Update:** you sent through the three photos downloaded from the links above. All
three are now embedded directly in the slides (replacing their placeholder boxes),
bordered in the deck's accent colour with the credit line underneath: the Blue
Marble full-disk view (deck 01, "The bigger picture"), the West Africa anvil and the
Borneo/Maritime Continent thunderstorms (deck 04, "Anvils, seen from space", side by
side). The Maritime Continent photo arrived at 2128x1416 (~2.0MB) -- resized to
1400x931 (~220KB) before embedding, visually identical at slide scale. Package is now
about 11.6MB.

## Sixth pass: motivation and quality -- you asked me to keep improving it, make it
really good, and motivate students to do good work.

I went back through the notebook, the handbook and all six decks and answered three
questions directly rather than assuming: is it clear what to do each week (yes, and
better now with "Today's plan"); is the material clear enough to self-teach after a
missed session (yes -- the notebook's own CORE/EXPLORE/OPTIONAL markdown and the
handbook are genuinely complete); and is it worth actually coming to tutorial (weaker
answer -- none of the six decks had a single peer-interaction or accountability prompt,
so the only real reason to attend was live troubleshooting and the two clinic weeks).
This pass targets that gap plus general motivation, without inventing a new formal
activity (a joint/practice-talk idea turned out to belong to a different semester, so
it's intentionally not included here):

- **A "why this matters" section, new in handbook §1.** Real-world ENSO stakes --
  drought/flood risk shifts, hurricane-season effects, and the 2015-16 global coral
  bleaching event -- framed as "the same signals forecasters actually watch," not just
  an academic exercise. Mirrored as a strengthened intro sentence on deck 01's "The
  bigger picture" slide, and as one added sentence each in the notebook's opening cell
  and its STOP cell.
- **A worked weak-vs-strong paragraph example, new in handbook §8** ("What this
  looks like in practice"). Two versions of the same (explicitly labelled illustrative)
  result, side by side, so students see concretely -- not just described in the abstract
  Claim/Evidence/Interpretation/Qualification diagram -- exactly what separates a 40-50%
  paragraph from a 70%+ one.
- **Light peer-accountability prompts** added to one slide each in decks 02, 03, 04 and
  05 (e.g. "show a neighbour your best figure without commentary -- can they tell what
  it shows?"). Cheap, one line each, but they give a concrete reason to be in the room
  with other people rather than reading the deck alone -- and the figure one doubles as
  a genuinely useful writing-clarity check.
- **Filled in every missing speaker-note timing.** The slides added in the fifth pass
  (Today's plan, the bigger-picture image, the marking-scheme slide, the deep-convection
  schematic, the anvil photos, the marking-scheme recap) had no timing notes at all --
  a real gap for a tutor prepping from notes. Added ~1 min each and recomputed every
  deck's total against the 50-minute slot; updated the explicit hands-on-budget line in
  deck 01 and every "Timing (50 min)" line in `Canvas_weekly_plan.md` to match. Tutorial
  6 (Writing workshop) is now the tightest at 17 min of direct teaching, still flagged
  as the one to watch.
- File sizes essentially unchanged (~11.6MB) -- this pass was almost entirely text and
  timing, deliberately, rather than more images or slides.

**Note for future passes:** Paul mentioned a "joint talk" -- an ungraded practice
presentation that used to help students get their story sorted before writing. On
reflection he confirmed that activity belongs to semester 2, not this 8-tutorial
package, so it's deliberately NOT included here. Don't re-add it to this package
without Paul asking again.

## Seventh pass

Paul reviewed deck 01's Walker-circulation schematic and deck 02's area-weighting
schematic as a domain expert and caught two real errors, plus asked for substantially
more guidance on reporting numbers and figure captions.

- **Fixed the Walker-circulation schematic (`walker_course_schematic.png`, deck 01,
  "Why should the clouds change?").** The previous version implied the rising/sinking
  sides of the Walker cell swap between La Niña and El Niño. Paul's correction: they
  don't -- the Maritime Continent/tropical western Pacific is *always* the rising
  branch and the eastern Pacific *always* the sinking branch. What changes is the
  *strength* of that circulation (La Niña strengthens it; El Niño weakens it, as warm
  water and convection partially shift toward the central/eastern Pacific), plus the
  fact that any extra cloud that shows up further east during El Niño is usually thin,
  not deep convection. Rebuilt the schematic with fixed geometry across both panels
  (same rising/sinking sides), La Niña shown as a stronger version of the loop and El
  Niño as a weaker one, and a clearly secondary, deliberately modest "sometimes more
  cloud near S. America too -- usually thin" annotation off to the side rather than a
  second rising branch. Also rewrote the slide's bullet text, which had the same
  "rising region shifts eastward" error, and re-embedded the corrected image into the
  slide (zip-level media swap, matching the box aspect ratio so nothing gets stretched).
  Also updated the same claim where it appeared a second time, buried in a "strong
  paragraph" worked example in handbook §8 -- it was making the identical physics error
  in a different place.
- **Fixed a real rendering bug in deck 02's area-weighting schematic** (slide 3, "Why
  area weighting matters"). Paul flagged the Earth/sphere picture as elliptical with
  overlapping red text -- both were genuine bugs, not just a description issue: the
  picture's embedded image had a different aspect ratio than the PowerPoint picture box
  it sat in, so PowerPoint was silently stretching it, turning the schematic's true
  circle into a visible ellipse; separately, the "radius = cos(60°)" label sat close
  enough to the circle's edge to overlap it. Fixed both: resized the picture box to
  match the image's actual aspect ratio (no more stretch), and moved both radius labels
  out to open space to the right of the circle with a thin leader line back to their
  point, so neither can ever overlap the circle again.
- **Added substantially more guidance on reporting numbers and figure captions**,
  per Paul's request. New in handbook §8: "Reporting numbers well (and how not to)"
  (precision vs. false precision, stating baselines/sample sizes, not overclaiming
  significance, correlation vs. causation, and a "common mistakes" list covering
  cherry-picking, unanchored percentages, and unsanity-checked numbers) and "What
  makes a figure caption adequate" (a five-point checklist plus a vague-vs-adequate
  caption pair, building on the existing attribution-line guidance). Reinforced lightly
  on the slides too: deck 06's "Write around evidence" (Tutorial 6) gained one more
  bullet pointing at the new checklist, without adding a new slide or eating into that
  deck's already-tight 50-minute budget.
- File sizes essentially unchanged; no slide counts changed in any deck.
