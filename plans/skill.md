---
name: verify-before-done
description: Use this whenever writing, editing, or refactoring code in any language or repo. Enforces grep-before-edit, run-real-tests-before-claiming-done, read-full-output, and root-cause-before-patch discipline. Trigger it for any change that touches a function/class/field signature, moves code between files, or when a test/build fails — not just for "big" refactors.
---

# Verify Before Done

This skill exists because a specific, repeatable failure pattern kept
happening: a change looked correct, compiled, and got reported as fixed —
and then broke almost everything, or quietly introduced a new bug that sat
undetected until much later. Every rule below traces back to a real bug from
a real session. Follow all of them, every time, not just when something
feels risky.

## The failure pattern this skill prevents

1. Agent moves/renames/removes a symbol (a class, a field, a function).
2. Agent updates the one place it was looking at.
3. Agent does **not** search the rest of the codebase for other usages.
4. Agent runs a quick syntax check or a narrow test, sees no error, and
   reports the change as done.
5. In reality, some other file still references the old symbol. It breaks
   at runtime, not at edit time — often much later, often for someone else.

A close cousin: the agent "fixes" a crash, then declares success because
the crash is gone — without noticing that the crash was masking two or
three *other* bugs that only become reachable once the crash stops
happening. Fixing bug #1 and stopping there is not finishing the job; it's
finding bug #2 for someone else to hit later.

## Hard rules

### 1. Before editing: find every usage, not just the one in front of you

Before renaming, removing, or changing the signature/meaning of anything
that isn't purely local (a class, a public function, a model field, a
config key, an exported constant):

- Grep the **entire repo** for the old name, not just the file you're
  editing and not just the directory it lives in. Include tests,
  scripts, migrations, admin/reporting code, and background jobs —
  these are exactly the places that get missed because they aren't in
  the main request/response path.
- If the symbol is a class or function, also grep for the module path it
  used to live in (`from old.module import X`) — moving code between
  files is the single most common source of stale imports.
- If it's a model field, grep for the field name as a bare word, not
  just `object.field_name`, since it can appear in `getattr`, format
  strings, serializers, or raw SQL.
- Update **every** match before considering the change complete. Do not
  update "the ones that matter" and assume the rest are dead code —
  prove it by grepping, not by guessing.

After editing, grep again for the old name. Zero hits, or every
remaining hit is a deliberate historical reference (a comment, a
changelog) — otherwise you're not done.

### 2. A green build or a clean import is not verification

Syntax checks, linters, and "it imports without error" tell you the code
is *parseable*, not that it *works*. They catch none of the bugs that
actually cause incidents: wrong field name, broken session/transaction
handling, off-by-one business logic, double-counted side effects.

Before saying a change is fixed:

- Run the actual test suite (or the actual application, exercising the
  actual code path) — not a subset picked because it seemed related, not
  just the file you touched. Run the whole thing unless it's genuinely
  too large to run in full, in which case run the largest slice you
  reasonably can and say explicitly which parts you didn't cover.
- Read the **full** output, not just the pass/fail count. A drop from
  166 failures to 1 failure is real progress — but that 1 failure still
  needs to be read, understood, and either fixed or explicitly called
  out, not glossed over because the number looks good.
- If you can't run the tests yourself (no environment access), say so
  plainly instead of asserting the fix works. "I've made this change,
  please run your test suite to confirm" is honest. "This fixes it" when
  you haven't run anything is not.

### 3. When a fix reveals new failures, that's the job continuing, not a new problem

If fixing one bug causes previously-hidden errors to surface (new test
failures, new stack traces, behavior that only now executes), do not
treat this as unrelated or as scope creep. It usually means the first bug
was masking others. Standard sequence:

1. Fix the root cause you found.
2. Re-run everything.
3. For every failure that remains, check: is this the *same* root cause
   showing up in another place (same fix should cover it — did you miss
   a usage per rule #1?), or is it a *genuinely different, previously
   unreachable* bug that the first fix exposed?
4. Diagnose each remaining failure on its own terms — read its actual
   traceback, don't assume it's "probably the same thing."
5. Repeat until the remaining failures are either fixed or are pre-
   existing/out-of-scope issues you've explicitly flagged to the user.

Don't stop at "most things pass now." Stop when you've accounted for
every failure, one way or another.

### 4. Before patching, find the root cause — don't patch the symptom

If a fix requires touching internals you don't fully understand yet (a
framework's session/transaction lifecycle, an event system, a concurrency
model), stop and read enough of it to explain *why* the bug happens before
writing the fix. A fix based on "this pattern usually works" without
understanding the actual mechanism tends to trade one bug for a subtler
one (e.g., silently reusing a shared object across a boundary it wasn't
meant to cross).

Concretely:
- Reproduce the failure in isolation first (smallest test/script that
  triggers it) and read the full traceback, not a truncated summary.
- Identify the specific line and mechanism responsible, not just the
  error message's surface symptom.
- Only then write the fix, and explain in a comment or commit message
  *why* it works, not just *that* it works.

### 5. When behavior changes after a fix, decide deliberately whether the bug is in the code or in the test/data — don't reflexively "fix" whichever is closer

Sometimes a previously-passing test starts failing once a real bug stops
being masked. Before changing anything, work out which side is actually
wrong:

- Check invariants that reflect real-world rules (uniqueness constraints,
  type constraints, documented business rules). If the test/data violates
  one, the test/data is wrong, not the constraint.
- Check whether the production code path being exercised is *authoritative*
  or *incidental*. If two independent code paths do the same job (e.g.
  two different places that both bill the same thing, or two listeners
  that both react to the same event), that's usually the real bug — not
  either individual line — and it needs a decision about which path
  should own that responsibility, not a quick patch to either one.
- State the reasoning, not just the diff. "I changed X because Y models
  a real invariant and Z was accidentally violating it" is verifiable by
  someone else; "I changed X to make the test pass" is not.

### 6. Watch specifically for these smells, because they've each caused real incidents

- **Split-brain refactor**: a class/field exists in two places (old and
  new), and only one was fully cut over. Search for duplicate
  definitions of the same concept before assuming a move is complete.
- **Shared mutable state across a boundary it shouldn't cross**: reusing
  a session/connection/transaction object inside code that's already
  mid-operation on that same object (e.g. calling `.flush()` from inside
  a flush event handler). If you're inside a callback/event/signal
  handler, assume the enclosing object is in a special state and verify
  before reusing it directly.
- **Unintentional double side-effects**: an automatic hook (event
  listener, signal, cron job) and an explicit code path both perform the
  same real-world action (billing, sending a notification, writing a
  ledger entry) with no de-duplication between them. When you add an
  automatic hook, explicitly check whether an existing explicit path
  already does that job.
- **Reused identifiers that model distinct real-world entities**: two
  records sharing a value that a uniqueness constraint says should be
  unique (e.g. two payments with the same receipt number) is a data bug,
  not a coincidence — the fix is to give them distinct values that
  reflect the actual scenario, not to weaken the constraint.

## Before reporting "done", answer these explicitly

- [ ] Did I grep the whole repo for every symbol/field I changed, moved,
      or removed — not just the file I was editing?
- [ ] Did I re-grep afterward and confirm no stale references remain?
- [ ] Did I run the real test suite (or real app), not just a syntax or
      import check?
- [ ] Did I read the full output, including the exact failure count and
      every remaining failure's actual traceback?
- [ ] If new failures appeared after my fix, did I diagnose each one
      individually rather than assuming they're related or pre-existing?
- [ ] For any remaining failure, have I determined whether it's a product
      bug or a test/data bug, with reasoning tied to an actual invariant
      or design decision — not a guess?
- [ ] Would someone re-reading my diff understand *why* it's correct, not
      just *that* it changes some lines?

If any box is unchecked, the work isn't verified yet — say so, rather than
reporting success.
