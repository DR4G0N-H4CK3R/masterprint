# Validation record

Measured in this environment on 29 October 2026 (Asia/Kolkata).

- Two complete real-image runs: minutiae baseline and ORB comparison, seed 42,
  300 original fingerprints, person-disjoint protocol.
- A second primary run reproduces all saved arrays and evaluation metrics exactly.
- Seven unit tests pass, covering threshold ties, no-feature images, single-class
  metrics, crop dimensions and cluster bootstrap edge cases.
- Both result directories pass independent checks of image hashes, disjoint
  candidate people, calibration FMR and saved selection arithmetic.
- Primary-model local image inference executes successfully.
- Every delivered Python file passes syntax compilation.
- The 20-slide PPTX contains editable text, tables and charts. Slide previews were
  visually inspected and the PPTX package passed structural and chart checks.
- PowerPoint desktop and Windows execution were not available in this environment.

AUC and calibration error are research measures on same-image crops. These
checks establish pipeline execution and result consistency, not production
biometric security.
