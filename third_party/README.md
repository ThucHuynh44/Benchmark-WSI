# Frozen upstream snapshots

`upstream/` contains byte-identical files copied from the pinned revisions of
LWSR, MICIL, and QPMIL-VL. These files exist only for provenance and algorithm
comparison. ConSlide runtime code must not import them.

Each project directory contains a `SOURCE_MANIFEST.json` with the canonical
repository URL, full commit SHA, and SHA-256 digest for every copied file.
Do not edit files listed by a manifest. Make integration changes only in the
active ConSlide implementation and document the resulting differences there.

MICIL and QPMIL-VL carry additional `INTERNAL_RESEARCH_ONLY.md` notices. Review
the upstream license or absence of a license before copying, modifying, sharing,
or publishing any derived implementation.
