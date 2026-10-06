# Specification

This folder will hold the full specification of evoparam: scope, modules, data formats,
public interface, validation datasets and release plan.

The current skeleton (version 0.1) fixes only the four-block layout
(`io`, `detect`, `clean`, `param`) and one working path through it.

Inherited modules to review during the specification:

- `objs` (`evoparam`, `evoparamGroup`): class names and settings.
- `bands`: named time windows; may become part of the results interface.
- `sim`: simulation of evoked responses; depends on neurodsp for noise.
- `plts`, `core/reports`: plots and PDF reports.
- `core/io`, `utils/io`: saving and loading model results.
