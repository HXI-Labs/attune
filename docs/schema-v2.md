# Attune schema v2

Schema v2 fixes two semantic problems that cannot be repaired compatibly in
v1: fake whole-utterance timestamps and invented numeric values for unavailable
signals.

- `temporal_scope=localized` requires start/end timestamps.
- `temporal_scope=utterance` forbids timestamps and style word boundaries.
- Affect dimensions explicitly carry `available`; unavailable dimensions use a
  null value and zero confidence.
- Audio-quality estimates identify calibrated, heuristic, or unavailable
  evidence.
- Affect abstention requires a reason and null top label.

`attune.schema.migration.migrate_v1_to_v2` converts validated v1 objects. A
legacy `0..duration` annotation becomes utterance scope; other spans remain
localized. The migration never promotes placeholder quality values.

JSON remains authoritative. `render_xml_v2` accepts only a validated v2 object,
and trusted-channel packaging keeps spoken text separate from model metadata.
