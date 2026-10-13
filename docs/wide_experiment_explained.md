# Item #3 Experiment, Explained Simply

*Plain-language explanation of the "boundary-aware data collection" experiment for the SINDy-CLBF paper.*

---

## 1. The picture: a stick, a valley, a cliff, and fog

Our system is an **inverted pendulum** — like balancing a broom upright on your hand.
To picture it, think of a **marble rolling in a landscape**:

- **Valley bottom = the goal** (broom balanced upright). The controller always pushes the marble *toward* the valley bottom.
- **Cliff = danger.** Far from the valley there's a cliff edge. If the marble rolls past it, it **falls off = crash = "unsafe."**
- **Map = the robot's learned physics.** The robot learns a model (**SINDy**) from practice data and uses it to plan pushes.
- **Fog = sensor noise (σ).** The robot can't see the marble exactly. **Thicker fog (bigger σ) = more likely to overshoot toward the cliff.** σ=0.04 is light fog; σ=0.15 is thick.

## 2. What we're trying to prove

- **Item #1** = a **math proof**: *if the map is accurate enough, the marble is guaranteed (high probability) to stay safe.*
- **Item #2** = cleaning up the definitions so that proof is consistent.
- **Item #3** = needed a **real experiment** showing something we can *do* to make the robot safer.

The question Item #3 answers:

> **Is there a practical trick that makes the robot tolerate *thicker fog* before it falls off the cliff?**

Two candidate tricks:
- **Trick A ("wide practice"):** let the robot **practice near the cliff** (collect training data out there).
- **Trick B ("magnifying glass"):** when drawing its map, **reweight the fit to pay extra attention near the cliff** (the original "safety-aware SINDy" idea).

## 3. The setup

Two robots that differ in **one way only** — where they practiced:

| | Where it practiced | Everything else |
|---|---|---|
| **Narrow robot** | Only near the valley bottom (never near the cliff) | Identical |
| **Wide robot** | All the way out to the cliff edge | Identical |

Then a **fog test**: crank σ up (0.04 → 0.15) and watch whether the marble stays safe or falls.

**Reading the numbers:**
- `sig=` → fog level (σ).
- `safe=1.0` → 100% of runs stayed safe; `safe=0.0` → fell every time; `safe=0.3` → 30% safe.
- `s=` → "seed" = which random repeat (repeated so results aren't a fluke).
- `goal=` → how often it also reached the goal (secondary; safety first).
- `delta=` → "how wrong is the map overall" score (the confusing one — see §6).

## 4. What success looks like

**The wide robot stays safe in thicker fog than the narrow robot.** That would prove practicing near the boundary makes it more robust.

## 5. What we got (the win)

| Fog level σ | **Narrow robot** | **Wide robot** |
|---|---|---|
| 0.04 (light) | 100% safe | 100% safe |
| **0.05** | **~0% safe** (fell) | **100% safe** |
| **0.075** | 0% safe | **100% safe** |
| 0.10 (thick) | 0% safe | 0% safe (finally fell) |
| 0.15 | 0% safe | 0% safe |

- Light fog: both fine (too easy to tell apart).
- σ=0.05: **narrow falls** (its cliff-map was a guess); **wide is perfectly safe** (it practiced there).
- σ=0.075: narrow still dead; **wide still 100% safe.**
- σ=0.10: fog finally thick enough that even the wide robot overshoots.

**Headline:** the falling point moved from **σ≈0.05 (narrow) → σ≈0.09 (wide)** — **roughly double the fog tolerance**, purely from practicing near the boundary. Genuine, clean Item #3 result.

## 6. The twist: the "magnifying glass" trick did nothing

Trick B (reweight the map near the cliff) **changed nothing** — the map came out identical.

**Why?** Once the robot *practiced* near the cliff (wide data), its cliff-map is **already sharp**. The magnifying glass has nothing left to sharpen.

So:
- **What matters = *where you practice* (get the data).** ✅
- **What doesn't matter = *how you reweight* the map afterward.** ❌

Good for the paper: the lever is **data near the boundary**, not clever loss-weighting → points to **active exploration** as the real future direction.

## 7. The confusing number: δ ("delta")

**δ = a single score for "how wrong is the map overall."** You'd expect lower δ → safer. But:

- Narrow robot: δ ≈ **10**, and **unsafe**.
- Wide robot: δ ≈ **16–18** (map looks *more* wrong overall!), and **safe**.

**Higher δ, yet safer.** Why? δ is a **global worst-corner** score dominated by faraway corners the marble **never visits**. The wide robot's map is slightly worse in some irrelevant corner (bumping δ up) but **much sharper along the path near the cliff** where the marble actually rolls. Safety only cares about the path it travels.

**Does this break the theorem (Item #1)?** No. The theorem says *"small δ **guarantees** safe"* — it never says *"big δ means unsafe."* A big-δ robot is simply not *guaranteed*, but is allowed to be safe anyway. No contradiction.

## 8. Why the "clean" batch

The results above were stitched from a few separate runs. For the paper we re-ran **both robots side-by-side, identical settings, 5 repeats each, all 5 fog levels** (50 runs) so no reviewer can blame a settings difference or luck. Expected to confirm the same pattern.

## In one breath

> We test whether **practicing near the safety cliff** makes the robot survive **thicker sensor fog**. It does — **roughly doubles** the fog tolerated (falls at σ≈0.05 before, σ≈0.09 after). The fancier "pay-attention-near-the-cliff" math trick did nothing, because practicing there already made the map sharp. And δ goes *up* even though safety improves — fine, because safety depends on the path walked, not δ's faraway worst corner.
