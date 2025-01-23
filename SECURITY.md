# Security Policy

## Supported versions

densekit is pre-1.0. Security fixes are applied to the latest release on the
`main` branch only.

| Version | Supported |
| --- | --- |
| 0.1.x | :white_check_mark: |
| < 0.1 | :x: |

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Instead, use
GitHub's private [security advisory][advisory] form for this repository, or open
a minimal private report there.

Include, if you can:

- a description of the issue and its impact,
- steps or a short snippet to reproduce,
- affected version(s).

I aim to acknowledge reports within a few days and to ship a fix or mitigation
as soon as it is practical.

## Scope notes

densekit is offline-first: the core makes no network calls and downloads no
models. The most relevant surfaces are deserialization (`load_index` uses
`allow_pickle=False` by design) and any input parsing in the CLI.

[advisory]: https://github.com/ivyw960/densekit/security/advisories/new
