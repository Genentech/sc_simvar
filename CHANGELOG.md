# Changelog

## v0.3.1
- Fix: replace compute_weights hotspotsc import with sc_simvar._lib

## v0.3.0
- Enhancement: increased speedup of Rust code dramatically using Github Co-Pilot with Claude, now at 3x
- Enhancement: added improved benchmarking and results
- Change: removed unused Rust code

## v0.2.0
- Change: updated tools for managing and building the project
- Change: transcribed remaining hotspotsc functions to local functions (some will be moved to Rust in a future release)
- Change: updated Rust crate versions to latest, modified code to work with them

## v0.1.1
- Change: building with manylinux container directly resulted in a very slow package we now build with maturin manylinux container which seems to have fixed the issue

## v0.1.0
- Implements all of the functionality of the ``Hotspot`` class in the ``SCSimVar`` class.