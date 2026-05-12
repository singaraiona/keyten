# K9 Spec — Implementation Roadmap and Type System Audit

## Source

Authoritative reference: <https://estradajke.github.io/k9-simples/k9/index.html>
(complete K9 language manual by John Estrada, build 2023.03.09). The
index lists 21 sub-pages: `Introduction`, `Verb`, `Adverb`, `Noun`,
`List`, `Dictionary`, `User-Functions`, `Expression`,
`Named-Functions`, `Knit-Functions`, `I_002fO`, `FF`, `Tables`,
`kSQL`, `System`, `Control-Flow`, `Temporal-Functions`, `Errors`,
`Examples`, `Benchmark`, `Conclusion`.

This doc covers **type-system corrections needed now** plus the
**phased implementation roadmap for the rest of the spec**. K9 is
genuinely large; treat the roadmap as a multi-session arc, not a
single session.

---

## Part 1 — Type system audit (immediate)

The current `keyten/src/kind.rs` was authored against an informal K
reference, not the K9 spec. There are concrete divergences from K9
authoritative text (Noun.html), especially in temporal types.
Audit table below.

### K9 authoritative type table (from Noun.html § 4.1)

| K9 letter | K9 type name | Atom example | Vec letter | Storage | Notes |
|---|---|---|---|---|---|
| `b` | boolean | `1b` | `B` | 1 byte | values `0` `1` |
| `c` | character | `"a"` | `C` | 1 byte | ANSI |
| `g` | int | `2g` | `G` | **1 byte unsigned** | range 0–255 |
| `h` | int | `2h` | `H` | **2 byte unsigned** | range 0–65535 |
| `i` | int | `2` | `I` | **4 byte unsigned** | default for positive ints |
| `j` | int | `2j` | `J` | **8 byte signed** | default for negative / out-of-range positive |
| `e` | float | `3.1` | `E` | (see open question) | default float literal |
| `f` | float | `3.1f` | `F` | (see open question) | explicit float |
| `n` | name (symbol) | `` `abc `` | `N` | 8-byte packed | ≤8 chars |
| `d` | date | `2020.06.14` | `D` | 4 byte | **days since 2001.01.01** |
| `s` | time-second | `12:34:56` | `S` | (open q.) | time-of-day |
| `t` | time-millisecond | `12:34:56.123` | `T` | (open q.) | time-of-day |
| `u` | time-microsecond | `12:34:56.123456` | `U` | (open q.) | time-of-day |
| `v` | time-nanosecond | `12:34:56.123456789` | `V` | (open q.) | time-of-day |
| `S` | datetime-sec | `2020.06.15T12:34:56` | (— scalar/list distinction unclear) | 8 byte? | epoch 2001.01.01 |
| `T` | datetime-ms | `2020.06.15T12:34:56.123` | | 8 byte? | |
| `U` | datetime-us | `2020.06.15T12:34:56.123456` | | 8 byte? | |
| `V` | datetime-ns | `2020.06.15T12:34:56.123456789` | | 8 byte? | |
| `L` | mixed list | `(3;3.1;"b")` | | varies | non-uniform list |

> **Note on uppercase confusion**: K9 uses **uppercase letters for both
> uniform-vector codes AND for datetime atom codes**, depending on
> context. `B` is a boolean vector but `S` is a datetime atom (sec).
> This is a K9-specific convention different from kdb+/q where
> uppercase always means vector. Verify against more authoritative
> source before committing to it.

### Current keyten kinds vs K9

| keyten | code | storage | K9 equivalent | Status |
|---|---|---|---|---|
| `Bool` | 1 | u8 (1 B) | `b` | OK (storage byte vs bit-pack is implementation choice; K9 doesn't bitpack) |
| `U8` | 4 | u8 (1 B) | `g` | OK shape — but rename for K9 readability (see below) |
| `I16` | 5 | i16 (2 B) | `h` (**u16**) | ⚠️ **signedness mismatch** |
| `I32` | 6 | i32 (4 B) | `i` (**u32**) | ⚠️ **signedness mismatch** |
| `I64` | 7 | i64 (8 B) | `j` | OK |
| `F32` | 8 | f32 (4 B) | `e` (?) | ⚠️ **storage size unverified** |
| `F64` | 9 | f64 (8 B) | `f` | OK |
| `Char` | 10 | u8 (1 B) | `c` | OK |
| `Sym` | 11 | i64 (8 B packed) | `n` | OK |
| `Date` | 14 | i32 (4 B) | `d` | ⚠️ **epoch undocumented; K9 says 2001.01.01** |
| `TimeS` | 19 | **i32** (4 B) | `s` | ⚠️ storage unverified for K9 |
| `TimeMs` | 20 | **i32** (4 B) | `t` | ⚠️ storage unverified for K9 |
| `TimeUs` | 21 | i64 (8 B) | `u` | OK if K9 uses 8 B here |
| `TimeNs` | 22 | i64 (8 B) | `v` | OK if K9 uses 8 B here |
| `DtS` | 23 | i64 (8 B) | `S` | ⚠️ **epoch undocumented; K9 implies 2001.01.01** |
| `DtMs` | 24 | i64 (8 B) | `T` | ⚠️ same |
| `DtUs` | 25 | i64 (8 B) | `U` | ⚠️ same |
| `DtNs` | 26 | i64 (8 B) | `V` | ⚠️ same |
| `List` | 0 | 8 B (`RefObj`) | `L` | OK |
| `Dict` | 99 | 8 B (`RefObj`) | (no K9 letter — separate) | OK |
| `Table` | 98 | 8 B (`RefObj`) | (no K9 letter — separate) | OK |

### Concrete corrections needed

**P1 — Critical and certain (temporal):**

1. **Document `Date` epoch** as `2001-01-01` in `kind.rs` doc comment.
   `RAY_DATE_EPOCH = 2000` in rayforce (`src/lang/cal.h`) — confirms
   the K family uses 2000-or-2001 epoch and Unix epoch is wrong.
   keyten currently has zero doc on the epoch; user code reading the
   raw i32 has no way to interpret it.
2. **Document `DtS`/`DtMs`/`DtUs`/`DtNs` epochs** the same way. Likely
   the same 2001-01-01 midnight UTC anchor, scaled by sec/ms/us/ns.
3. **Add a `temporal` module** with conversion utilities:
   - `date_from_ymd(y, m, d) -> i32`
   - `date_to_ymd(days: i32) -> (i32, i32, i32)`
   - `time_from_hms(h, m, s) -> i32 / i64 depending on granularity`
   - `dt_from_ymdhms_ns(...) -> i64`
   - Round-trip tests.

**P2 — Storage and signedness verification (research, then fix):**

4. **Verify K9 `s`/`t`/`u`/`v` storage**: K9 spec lists letters but
   no explicit byte counts. keyten has i32 for s/t, i64 for u/v —
   plausible (i32 fits 1 day in ms = 86,400,000 ≤ 2³¹; doesn't fit
   in us or ns). Need to test against K9 reference impl if available,
   or pick the storage that's correct by capacity and document.
5. **Verify K9 `e` float storage**: doc lists both `e` (default,
   bare `3.1`) and `f` (explicit `3.1f`) as "float" without sizes.
   Two interpretations:
   - `e` = f32, `f` = f64 (K convention from kdb+/q)
   - `e` = `f` = f64, suffix is parse-time syntax only
   Need to confirm. If the first, keep F32. If the second, drop F32.
6. **Verify K9 `g`/`h`/`i` signedness**: doc says explicitly
   "unsigned" for all three. K convention in kdb+/q has them signed.
   The "K9-specific unsigned" is an unusual choice. If confirmed,
   change `I16` → `U16`, `I32` → `U32` (keep `I64 = j` signed).
   This changes the kernel set: `(u16 + u16) -> u16` etc., wraparound
   semantics, null sentinel choices.

**P3 — Letter-code mapping (parse and display):**

7. **Add a letter-code accessor** on `Kind`:
   ```rust
   impl Kind {
       pub fn letter_atom(self) -> char { /* b c g h i j e f n d s t u v */ }
       pub fn letter_vec(self) -> char { /* B C G H I J E F N D S T U V L */ }
       pub fn from_letter_atom(c: char) -> Option<Kind>;
   }
   ```
   Used by `@` (type accessor) and by parser for typed literals
   (`37h`, `3.1f`, etc.).

### Open research questions

These need authoritative answers before P2 corrections land. Options:

- Build a small K9 reference workload using a public K9 binary (if
  one is available) and probe `@` on literals.
- Read Shakti / kparc source if licensed.
- Default to K convention (signed h/i, f32 e / f64 f) and document
  the deviation if K9 actually differs.

I recommend defaulting to K convention (signed, f32/f64 split) and
**marking the K9-unsigned interpretation as a future correction** if
testing shows it matters for compatibility. Most existing K code in
the wild treats these as signed.

---

## Part 2 — Full K9 spec implementation roadmap

What's actually in K9 vs what's in keyten today:

| K9 section | keyten state | Effort |
|---|---|---|
| **Noun** (types) | 17 of ~17 letter types declared but with corrections needed (see Part 1) | 1 session (corrections + tests) |
| **Verb** (~25 primitives) | 4 implemented (`+ - * %`), 1 partial (`!` monadic only) | 3-4 sessions |
| **Adverb** (6 forms) | 1 implemented (`/` over), 5 missing (`\`, `'`, `':`, `/:`, `\:`) | 2 sessions |
| **List** (uniform, mixed, indexing) | parser supports literal; ops sparse | 1 session |
| **Dictionary** | not implemented | 2 sessions |
| **Tables** | declared (Kind::Table) but no impl | 2-3 sessions |
| **User-Functions** (lambda) | **designed (v1.1 doc); not implemented** | 3-5 sessions per existing design |
| **Expression** | basic; needs idiom recognition | 1-2 sessions |
| **Named-Functions** (`in`, `each`, etc.) | not implemented | 2 sessions |
| **Knit-Functions** | not implemented | 1 session |
| **I/O** (`0:`, `1:`, etc.) | not implemented | 2-3 sessions |
| **FF** (foreign function interface) | not implemented (probably defer to v2.x) | 2-4 sessions |
| **kSQL** (select/update/delete on tables) | not implemented | 4-6 sessions |
| **System** (`\t`, `\v`, etc.) | partial (`\t \v \h \\ \p`) | <1 session for the rest (`\l` load, `\d` dir, `\w` workspace) |
| **Control-Flow** (`$[c;t;e]`, `:[..]`, etc.) | not implemented | 1-2 sessions |
| **Temporal-Functions** (`z.d`, `z.t`, accessors) | not implemented | 2 sessions |
| **Errors** | basic `KernelErr`; needs K-style error AST | 1 session |

**Total estimated**: **25–40 focused sessions**. Treating as roughly
**1 quarter of dedicated work** at typical session cadence.

### Suggested phasing

#### Phase 0 — Type system corrections (1 session — this work)

- All P1 corrections from Part 1 (temporal epochs documented;
  conversion utilities; tests).
- P3 letter-code mapping.
- P2 deferred until research lands.
- Update `kind.rs` doc to point at the K9 spec URL.

**Acceptance:** `kind.rs` round-trips letters↔Kind; date/time
conversion functions round-trip y/m/d ↔ days and h/m/s ↔ units;
existing tests still pass.

#### Phase 1 — Lambda (v1.1) implementation (3-5 sessions)

Per the existing `2026-05-12-lambda-v1-1.md` design. Lambda is the
substrate for adverbs over user functions and for v3 reactor work.

**Acceptance:** `{x*x}[5] → 25`; `{[x;y]x+y}[3;4] → 7`; juxtaposition
`f x` for monadic; tests + bench.

#### Phase 2 — Verbs (3-4 sessions)

Implement the missing primitive verbs. Performance-mandate applies:
each verb gets a `ChunkStep` impl with a parallel branch in the same
commit. Suggested order (impact-weighted):

1. **`#` count + take** — universal; takes is also slice-construction
2. **`,` enlist + concatenate** — list/vector building
3. **`@` type / index / apply** — already needed for letter codes
4. **`!` partial verb** — needs the dyadic form (`x!y` = mod) and
   dictionary constructor
5. **`=` equality + freq** — comparison kernel + dict-of-counts
6. **`<` `>` `~` comparison + match** — bool-producing kernels
7. **`&` `|` min/max + and/or** — same-type-as-input kernel
8. **`^` sort/cut** — sort is a major piece (parallel quicksort)
9. **`_` floor/drop** — float→int kernel + slice op
10. **`$` string convert / parse** — needs proper char-vector handling
11. **`?` unique / find** — hash-set kernel
12. **`.` value / dict** — eval + dict constructor

Per-verb pattern:
- `kernels/<verb>.rs` with `ChunkStep` for each kind pair
- `_async` dispatch entry
- Sync shim via `block_on`
- Parallel branch via `parallel_for_each_mut`/`parallel_reduce`
- Tests in `tests/parse_eval.rs`
- One commit per verb to keep history bisectable

#### Phase 3 — Adverbs (2 sessions)

`/`, `\` already partial. Add:
- `\` scan (running aggregate)
- `'` each
- `':` eachprior
- `/:` eachright
- `\:` eachleft
- Converge forms `/:` and `\:` when applied to a function only

Adverbs compose with verbs and lambdas — needs lambda support
(Phase 1).

#### Phase 4 — Composite types: Dict + Table (3-4 sessions)

- Dict: `keys!values` literal, `dict@key` lookup, `!dict`/`.dict`
- Table: `flip dict` constructor, `cols`/`!t`/`.t` access
- kSQL: separate Phase 6

#### Phase 5 — Control flow + system + errors (2 sessions)

- `$[c;t;e]` conditional
- `:[c;t;e;...]` cond
- Error AST type (catchable errors)
- `\l`, `\d`, `\w` system commands

#### Phase 6 — Temporal functions (2 sessions)

- `z.d`, `z.t`, `z.T` accessors
- `.h`, `.r`, `.s`, `.t`, `.u`, `.v` extractors on time values
- Time arithmetic (date + days, time + millis, etc.)
- Calendar functions if K9 has them

#### Phase 7 — kSQL (4-6 sessions)

`select`, `update`, `delete`, `exec`, `from`, `where`, `by`, `do`.
This is a major mini-language inside K9; deserves a dedicated phase.
Note: rayforce's optimizer pipeline (idiom rewrite, predicate
pushdown, partition pruning, projection pushdown, fusion, DCE) shows
the shape this can take when serious. For keyten we can start with a
direct execution model and add the optimizer later.

#### Phase 8 — IO + FF (3-5 sessions)

- File I/O: `0:` (text), `1:` (binary), `2:` (stderr/debug)
- Network: `\s` (sockets) in K9
- FF: `2:` overload for foreign function calling (defer — large)

#### Phase 9 — Knit-functions + named-functions (2-3 sessions)

`in`, `like`, `within`, `where`, `bin`, `find`, etc.

### Where v3 (reactor) fits

The reactor lands **after Phase 1 (Lambda)** because reactions ARE
lambdas with reagent metadata. It can land in parallel with Phases
2-9 as long as Lambda is done. Reagents (channels, timers, TCP) are
mostly tokio-stream wrappers per the earlier design discussion.

---

## Part 3 — Recommendation for the immediate session

Given the scope, I propose **doing Phase 0 in this session** (type
system corrections, with the temporal epochs documented and conversion
utilities landed) and treating Phases 1–9 as sequenced future work
each with its own focused session.

### What Phase 0 actually changes (concrete edits)

1. **`keyten/src/kind.rs`**:
   - Update module-doc to point at the K9 spec URL.
   - Document the **2001-01-01 epoch** in `Kind::Date` and the
     four `Dt*` variants. State the granularity unit explicitly.
   - Add `letter_atom()`, `letter_vec()`, `from_letter_atom()`
     methods on `Kind`.
   - Add a `kind_letter` test verifying round-trip.

2. **New `keyten/src/temporal.rs`** (~150 LOC):
   - `K9_EPOCH_YEAR: i32 = 2001`
   - `days_from_ymd(y, m, d) -> i32` with leap-year handling
   - `ymd_from_days(days: i32) -> (i32, i32, i32)`
   - Time-of-day helpers per granularity
   - DateTime composition: `dt_ns_from_date_and_time(d, t) -> i64`
   - Tests: known dates round-trip (2001-01-01 → 0, 2000-12-31 → -1,
     2020-06-14 → 7104, etc.)

3. **Defer signedness/float corrections** (P2) until the K9 reference
   binary or source is reachable. Document the assumption clearly so
   future contributors don't repeat the audit.

### What's left after Phase 0

Phases 1-9 follow the roadmap. Each phase is a focused session arc
with its own design doc if non-trivial. The Lambda v1.1 design
(2026-05-12-lambda-v1-1.md) is the next on-ramp.

### What this doc unblocks

- A reader (future-me, contributors, agents) can see the **full scope**
  in one place rather than discovering it piecewise.
- The **type-system fixes are unambiguous** — the temporal epoch is
  the single most important correction, and it's now documented.
- The **performance mandate** (CLAUDE.md) applies to every new verb,
  adverb, and kernel added in the roadmap — no implementation lands
  without its parallel/SIMD/cache analysis.

---

## Open follow-ups beyond this doc

- **K9 reference binary or source access** — without it, P2 questions
  (signedness, `e` storage, `s`/`t` size) can't be resolved
  authoritatively. Suggested next: try to obtain a Shakti binary,
  or correspond with Estrada / kparc.
- **K9 numeric type code scheme** — keyten uses 0–99; K9 docs only
  use letters. If K9 has internal numeric codes (likely, since q
  does), align to make `@` ergonomic. If not, keep keyten's scheme.
- **Cross-implementation conformance tests** — once a K9 reference
  is available, build a test corpus that exercises each verb /
  adverb / type on inputs whose results can be compared byte-exact.

---

## Live progress (updated 2026-05-12)

### Phase 0 — Type system corrections ✅ DONE
- `kind.rs`: 2001-01-01 epoch documented; K9 letter codes added.
- `temporal.rs`: Hinnant civil_from_days conversion; time-of-day helpers; dt composition.
- All round-trip tests pass.

### Phase 2 — Verbs: 15 of ~17 covered

| Verb | Monadic | Dyadic | Status |
|---|---|---|---|
| `+` | identity | add | ✅ both |
| `-` | negate | subtract | ✅ both |
| `*` | identity | multiply | ✅ both |
| `%` | sqrt (TODO) | divide | dyadic |
| `!` | til | mod (TODO) | monadic |
| `,` | enlist | concat | ✅ both |
| `#` | count | take | ✅ both |
| `@` | type | index/apply (needs λ) | monadic |
| `=` | freq (TODO) | equality | dyadic |
| `<` | asc-idx (TODO) | less-than | dyadic |
| `>` | desc-idx (TODO) | greater-than | dyadic |
| `~` | not | match | ✅ both |
| `&` | where (TODO) | min | dyadic |
| `\|` | reverse (TODO) | max | dyadic |
| `_` | floor | drop | ✅ both |
| `$` | string-convert | parse (TODO) | monadic |
| `?` | unique (TODO) | find (TODO) | — |
| `.` | value (TODO) | dict/apply (TODO) | — |
| `^` | sort (TODO) | cut (TODO) | — |

### Phase 3 — Adverbs: 2 of 6 covered
- `/` over — parallel reduce ✅
- `\` scan — parallel two-pass prefix sum (i64), sequential (f64) ✅
- `'` `':` `/:` `\:` — not started

### Substrate hardening done in parallel with verbs
- Performance mandate locked in `CLAUDE.md`.
- TSan re-run after every kernel-touching commit; clean across 12 parallel tests.
- Rayforce parity on `+/!1e8` (~37 ms median, commit `2a5c80a`).
- Test count: 66 parse_eval, 25+ lib unit, 12 parallel/TSan, 6 temporal.
