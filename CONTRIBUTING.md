# Contributing

Contributions and bug reports from the community are welcome!

## Future work

If time permits:

- Make test suite even more comprehensive.
- Move what remaining Python code that can be moved to Rust to Rust (most of the code in the src/sc_simvar/_not_yet_rust directory)
  - Code that relies on complex code from other libraries (like sklearn) will most likely remain Python

## Versioning

We try to abide by [Semantic Versioning](https://semver.org/).

- Bug fixes get a PATCH bump
- Back compatible changes get a MINOR bump
- API breaking changes get a MAJOR bump

Sometimes we may bump MAJOR because we feel like we, or the repo, deserves it.

## Setup & Testing

To develop this project you need to have [Rust](https://www.rust-lang.org/tools/install) and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

The `pyproject.toml` uses [Maturin](https://www.maturin.rs/) to build the Rust code, etc. so you can build and test this project in the usual way with uv:

```sh
uv sync
uv run pytest
```

If you make changes to the Rust code you will like have to run the command:

```sh
uv sync --reinstall-package sc_simvar
```

To get your latest changes into the `uv` `.venv` as Rust code changes aren't part of the [editable](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) install of the Python code.


## Docs

The docs are created using [mkdocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.