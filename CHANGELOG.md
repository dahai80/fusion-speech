---
doc_id: fusion-speech/changelog
product: fusion-speech
class: app
entry: service
doc_type: changelog
title: { zh: "变更日志", en: "Changelog" }
audience: engineer
status: draft
version: 0.1.8
updated: 2026-11-07
owner: "@dahai"
source_of_truth:
  - CHANGELOG.md
  - README.md
---

# Changelog

All notable changes to fusion-speech are documented here. The
milestone-by-milestone history (M1-M7, hardening passes) lives in README.md
("Status" section); this file tracks changes going forward in
Keep-a-Changelog format.

## [Unreleased]

## [0.1.8] - 2026-11-16

### Added
- Offline test CI (PR #10, fusion#2 CI retrofit): `tests` workflow running
  `ruff check` + `pytest` (no model load) with concurrency cancel grouped by
  ref, pip cache keyed on the manifest, and path filters so docs-only PRs
  skip tests. `docs` workflow gained the same concurrency guard.

### Fixed
- `test_f2_fail_closed_rejects_unverified_peer` handles the server closing
  before the client's write completes (Linux race): a reset/broken pipe is
  treated as the rejection, same as an immediate EOF.

## [0.1.7] - 2026-11-07

### Added
- Adopted the fusion docs-system v1.1 contract (grid position app x service):
  `docs/product-page.md`, `docs/get-started.md`, `docs/api-reference.md`,
  `docs/config-reference.md`, `docs/architecture.md`, `docs/troubleshooting.md`,
  `CHANGELOG.md`, 12-key frontmatter on in-scope markdown, and CI validation
  via `ci/validate_docs.py` gating markdown PRs (`.github/workflows/docs.yml`).

### Fixed
- `tests/test_wake_word.py::test_r2_download_off_loop` now skips with a cited
  reason when the optional `[audio]` extra is absent, instead of failing on the
  real openwakeword cold-import path (env precondition, not a code defect).
