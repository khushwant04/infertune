# Documentation

Start with the [design plan](plan.md) for the full picture, or jump to a topic.

## Design

- [Design plan](plan.md): the problem, five design decisions, architecture, data models, memory and performance models, adapters, search, milestones, risks and open questions.
  - [§2.6](plan.md): two errors the plan originally contained, and how measurement caught them.
  - [§9](plan.md): each milestone's acceptance criterion, fixed before the work started.

## Evidence

- [Project status](status.md): what is built, what is *measured*, and what remains. When this page and another disagree, this page is right.
- [A10 validation](gpu-validation-a10.md): the validation environment, the driver → CUDA → torch → vLLM constraint chain, and the M2 measurements.

## Operations

- [GPU runbook](gpu-runbook.md): the next validation session step by step. It covers `tune --execute` on hardware, the M3 throughput criterion, tensor parallelism over PCIe, and SGLang.

## Project

- [Changelog](../CHANGELOG.md)
- [README](../README.md)

## Writing docs

- Separate what is *built* from what is *measured*. A test against a fixture or a synthetic objective is not a hardware measurement, so say which one a number came from.
- Every number in a doc should be reproducible from a command in the repository or a measurement recorded here.
- Diagrams use Mermaid so they render on GitHub and stay diffable.
