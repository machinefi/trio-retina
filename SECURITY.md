# Security Policy

## Supported versions

Retina is on the `0.2.x` line. Security fixes are applied to the latest release
on the `main` branch.

## Reporting a vulnerability

Please report security vulnerabilities **privately** — do not open a public issue
for security problems.

Use GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/machinefi/trio-retina/security)
   of this repository.
2. Click **Report a vulnerability** and fill in the advisory form.

This opens a private channel with the maintainers. Please include:

- a description of the issue and its impact,
- steps to reproduce (a minimal example helps), and
- any affected versions you know of.

We will acknowledge your report, investigate, and keep you updated on the fix.
Once a fix is released, we are happy to credit you in the advisory unless you
prefer to remain anonymous.

## Trust model

Retina treats its pipeline configuration as **trusted operator input**. The
`WebhookSink` URL, the `JsonlSink` path, and the `video_frames` source (file,
RTSP, or URL) are taken at face value and used as given — Retina does not
sandbox or sanitize them. If you load a workflow JSON (`Pipeline.from_json`)
from an untrusted party, **validate those URLs and paths yourself** before
running it: apply a scheme allow-list (e.g. `https`, `rtsp`), and block
link-local / metadata addresses and `file://` URLs you did not intend to expose.

## Thank you

Thank you for helping keep Retina and its users safe.
