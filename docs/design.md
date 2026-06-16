# Design notes

Retina is the **perception encoder** of a world model. In the standard
decomposition (encoder → dynamics → decoder), Retina is the encoder — `s = Enc(x)`:
raw real-world signals → state. It produces *state*, not the dynamics or the
policy; we say "encoder," not "world model."

The full rationale — the dual symbolic + latent output, where Retina sits in the
world-model stack, the deep-vs-wide L1 axes, what it absorbed from DeepStream /
Holoscan / academia, and the hard layer-boundary rules — is the canonical
[**`DESIGN.md`**](https://github.com/machinefi/trio-retina/blob/main/DESIGN.md) in
the repository, kept there as the single source of truth so this site never drifts.
