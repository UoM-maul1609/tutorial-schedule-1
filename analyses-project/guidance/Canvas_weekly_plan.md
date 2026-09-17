# ENSO and cloud-cover mini-project: weekly Canvas plan

Every session is timetabled for **50 minutes**. Each week below budgets to 50 min
(content + hands-on + a 5-minute buffer); the full slide-by-slide breakdown is in each
deck's speaker notes (Presenter View / Notes). Two sessions carry a real timing risk on
top of the content itself — see the **Timing risk** callouts and the note at the bottom
of this page.

## 29 September — Project launch
**Slides:** Tutorial 1
**Guide:** Sections 1–4
**In class:** ERA5, ENSO, why clouds respond, launch Binder and run the first notebook sections.
**By the end:** notebook running; two interesting observations written down.
**Timing (50 min):** ~14 min science slides (ERA5, ENSO, Walker circulation) + ~25 min
hands-on (launch Binder, run to the first cloud figures) + 5 min settle/buffer + 6 min
housekeeping/framing (today's plan, the bigger-picture cloud-cover image, and the
marking-scheme pointer).
**Timing risk:** this is everyone's first Binder launch. A cold build can take several
minutes on its own — see "De-risking the 50 minutes" below.

## 6 October — Making sense of climate data
**Slides:** Tutorial 2
**Guide:** Section 5
**In class:** area weighting, seasonal cycle, climatology and anomalies.
**By the end:** students can explain what an anomaly means and why cos(latitude) weighting is used.
**Timing (50 min):** ~12 min science slides + ~30 min hands-on (climatology/anomaly
sections) + 5 min recap/buffer + 3 min framing (recap + today's plan + today's goal).

## 13 October — Finding the ENSO signal
**Slides:** Tutorial 3
**Guide:** Sections 3 and 5
**In class:** ONI, El Niño−La Niña composites, correlation/regression.
**By the end:** identify 2–3 interesting ENSO/cloud patterns.
**Timing (50 min):** ~14 min science slides (ONI, composite, correlation, discovering a
question) + ~29 min hands-on (the global-maps section) + 5 min buffer + 2 min framing
(recap + today's plan).
**Timing risk:** the maps section is the heaviest cell in the notebook (loads the full
lat/lon/time grid and draws three world maps with coastlines). It's also the first time
most students' notebooks will call Cartopy, which normally needs to download map outline
data on first use — see "De-risking the 50 minutes" below.

## 27 October — Choose your investigation
**Slides:** Tutorial 4
**Guide:** Section 6
**In class:** cloud levels, regional physics, optional autocorrelation/lag.
**By the end:** ONE research question + provisional 3–5 figure plan.
**Timing (50 min):** ~11 min science slides (cloud levels, deep convection + anvil
imagery, marine low cloud) + ~5 min written exercise (their research question, in their
own words, before hands-on starts) + ~25 min hands-on (regional analysis; autocorrelation
only if there's appetite) + 5 min buffer + 4 min framing (launch, today's plan,
narrowing checklist).

## 10 November — Analysis workshop
**Slides:** Tutorial 5
**Guide:** Sections 6 and 8
**In class:** adapt only the analyses needed for the chosen question.
**By the end:** 3–5 provisional figures + one result sentence per figure.
**Timing (50 min):** ~9 min framing/what-makes-a-good-figure + ~35 min hands-on
(this is a workshop — minimise talking, circulate) + 5 min buffer + 1 min milestone check.

## 17 November — Writing workshop
**Slides:** Tutorial 6
**Guide:** Sections 7–9
**In class:** Introduction, Methods, Results, Discussion, Conclusion; captions and attribution.
**By the end:** report skeleton + selected figures + provisional main conclusion.
**Timing (50 min):** ~17 min science-of-writing slides (this deck has the most content —
see the pacing note in its speaker notes if you're running behind) + ~26 min hands-on
(start the report skeleton with their own figures dropped in) + 5 min buffer + 2 min
framing (launch + today's plan).

## 24 November — Draft clinic
No new lecture. Students bring substantial draft material and final figures. Feedback
focuses on question → evidence → interpretation.
**Timing (50 min):** with no slides to pace against, the risk is running out of time
before reaching every student. For a class of N students, budget roughly `45 min / N`
per student and say so at the start (e.g. "~3 min each, so I can get to everyone — come
back if you want longer"); triage by circulating once quickly to see who's stuck before
doing deeper one-on-ones.

## 1 December — Final clinic
No new content. Resolve remaining scientific/figure/writing issues and revise.
**Timing (50 min):** same triage approach as 24 November. Since this is the last contact
point before submission, prioritise students who haven't had feedback yet over a second
pass with students you've already helped.

---

## De-risking the 50 minutes: Binder and Cartopy

Two things outside the slides can eat into class time if not handled in advance:

1. **Binder's first build is slow (minutes, not seconds).** BinderHub builds a fresh
   Docker image the first time anyone launches a given commit of the repo; every launch
   after that reuses the cached image and starts in seconds. **Launch the Binder link
   yourself 15–30 minutes before each session** (especially 29 September) so your
   students' launches hit the warm cache instead of triggering a rebuild.

2. **Cartopy downloads map-outline data on first use, which needs working internet from
   inside the Binder session.** This showed up directly while testing the refactored
   notebook: the default download host was unreachable from a locked-down network, and
   the maps section (13 October) is exactly where this first happens for most students.
   The fix is a `postBuild` file at the repo root (added — see `postBuild`) that
   pre-fetches the map outline data into the Binder image at build time, so students
   never depend on live network access to a third-party host during the session itself.
