# 09 — Distributed search

`honion search --listen` turns one machine into a coordinator that accepts
`honion worker` connections from other machines, so a fleet's GPUs add to
one search, one pattern set, and one `--out` directory. The design question
was never "how do we ship work to GPUs" — that part is easy. It is "does the
security model of [05](05-security-model.md) survive the work crossing a
network", and the answer drives everything below.

## The rule: the wire sees what the GPU already sees

The invariant `docs/05` defends locally — *no secret enters the device, no
key leaves it* — is what the protocol reproduces remotely:

* **Into the worker:** pattern text (configuration the operator would
  otherwise type at the coordinator's own keyboard) and, per epoch, the
  epoch's 32-byte *public* start points. The scalars behind them are drawn
  on the coordinator and never leave its process; the worker's input is
  byte-for-byte what a local GPU's would be.
* **Out of the worker:** `(thread, offset)` claims, a candidate count, and
  the device's self-reported name. A claim is a claim: the coordinator
  re-derives each candidate from its own scalars with `VerifiedKey::verify`
  — the same verifier a flaky local GPU's hits must pass — and only the
  coordinator ever calls `write_service_dir`.

A remote worker therefore knows strictly less than the local GPU already
knows, and a compromised worker can do exactly what a compromised local GPU
could do: waste time, or fabricate claims that fail verification. Nothing
more. The `--suffix` check stays host-side for the same reason: a worker
cannot filter on what it cannot see, so suffixes cost remote workers nothing
and are enforced identically (docs/04).

What is *not* provided is transport security. The wire carries no secrets,
but it does carry the patterns in cleartext, and a man in the middle can
drop connections, forge claims (they fail verification), or slow the search.
The honest boundary is *a network you trust not to waste your time*; TLS or
a wireguard tunnel is the answer if even that is too much.

## Protocol

One TCP connection per worker, `HON1`-framed binary (`crates/honion-cli/src/net/proto.rs`):
a four-byte magic that doubles as the version digit, a closed set of message
types, a payload length ceiling applied before any allocation, UTF-8 that
must actually be UTF-8, and start-point payloads that must be exact
multiples of 32 bytes. Every reader validates before it trusts; the worst a
hostile peer causes is a clean hangup.

```
worker ──Hello{device}──────────────▶  "here is what I am"
worker ◀─────────────Config{patterns, half, cadence…}
worker ──Ready{threads}─────────────▶  (threads sized after Config: only the
                                       config carries the batch width that
                                       thread sizing needs)
worker ◀─Epoch{id, points[N]}─┐
worker ──Report{id, examined,│ claims[(thread, offset)]}─┐  repeat
worker ◀─Epoch{id+1, …}──────┘   (the coordinator verifies in between)
worker ◀─Done──────────────────────  (--count met, or the operator stopped)
```

Launch cadence is the worker's own business: the coordinator states how
often it wants a report, and the worker reproduces the local loop's
self-tuning — one deliberately small first launch to measure the card,
every later launch sized from the measured rate. A 4090 and a Mac report at
their own natural rhythm over the same connection config.

## Failure semantics are asymmetric, on purpose

A *local* worker that dies stops the run: its failure means this machine's
search is damaged. A *remote* worker that hangs up is the ordinary case of
someone's laptop going to sleep: it is reported, and the remaining workers
keep searching. Every coordinator-side socket wait is bounded by a read
timeout, so a worker whose machine dies mid-epoch frees its thread; the
search's teardown joins every connection thread before reporting, so "done"
means no claim is still in flight.

Progress, `--count` and the checkpoint are the coordinator's: remote work
folds into the same combined rate line and the same `.honion-state`, so a
coordinator's `--resume` knows about every key the fleet wrote. A worker has
no state to lose; reconnecting means a fresh `Hello`, and the search is
memoryless anyway (docs/01).

## Topology

```
honion search --prefix carroll --out ./keys --listen 8121            # + own GPUs
honion search --prefix carroll --out ./keys --device none --listen 8121   # CPU-only coordinator
honion worker --coordinator gpu-farm.example:8121                    # rent a GPU
honion worker --coordinator 192.168.1.10 --device 1                  # its second card
```

Local devices and remote workers mix freely — a coordinator with
`--device all --listen` searches its own cards and the fleet at once. The
coordinator draws each connection's epochs independently (the same CSPRNG
independence that makes duplicate work between local devices a
2⁻¹⁹⁰ non-event), so no coordination protocol between workers is needed or
wanted. One worker process rents one device; a four-card box runs four
workers.

The thread-count bound is the wire: an epoch frame is capped at 64 MiB, so
connections may promise up to two million threads — four times what
`auto_threads` ever picks — which refuses nonsense, never hardware.

## What is deliberately not here yet

* **Secrets never flow, but neither do results in transit:** a worker never
  receives keys, so there is no result channel to secure. If a future mode
  wants "write to the worker's disk too", it starts a new trust analysis,
  not a patch to this one.
* **One epoch in flight per connection.** The local loop overlaps the next
  epoch's scalar draw with its GPU's launch; the remote loop draws *after*
  the previous report lands, costing the draw time (tens of milliseconds)
  against launch intervals of seconds. Two epochs in flight would erase
  that; it stays out until a measured case justifies the buffer's scalars
  living longer.
* **No worker authentication.** Authenticate the network (wireguard, or a
  coordinator inside the same private network), not the worker: an
  unauthenticated worker is already limited to wasting the coordinator's
  time.
* **No TLS.** Transport security would protect pattern text and public
  points. If the patterns themselves are sensitive to you, the right
  posture is a tunnel, which also buys authentication.
