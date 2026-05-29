# Release Checklist

- [ ] Tests pass with `python -m pytest`.
- [ ] `edge-inspect --help` works after installation.
- [ ] `python -m converter.cli --help` remains backward compatible.
- [ ] Sample commands in `examples/commands.md` are still accurate.
- [ ] README quickstart and limitations are up to date.
- [ ] No source model files are committed (`*.pt`, `*.pth`, `*.ckpt`).
- [ ] No large generated artifacts are committed.
- [ ] No TensorRT engines or plans are committed (`*.engine`, `*.plan`).
- [ ] CI uses only synthetic fixtures and does not require private real models.
- [ ] Changelog includes the user-facing changes.
