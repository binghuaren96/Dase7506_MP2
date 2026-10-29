# AI assistance and reuse disclosure

OpenAI Codex assisted with project planning, Python/NumPy implementation,
debugging, experiment execution, numerical and interface checks, analysis,
documentation, report drafting and preparation of the reproducible delivery entry. AI assistance
covered the candidate/group models, feature extraction, Bradley–Terry training,
cross-validation, limited parameter comparisons and resource measurement.
Claude also assisted with reviewing method ideas and updating the final
checkpoint/configuration from the earlier E3 model to the group ranker.

The final submission uses the supplied candidate solutions and fixed embeddings.
AI coding assistance did not add generated mathematical solutions, outside
training data, pretrained inference models or external model calls to the
selector. Candidate correctness labels were derived from the supplied train
references with the unchanged official parser. Validation was used repeatedly
for development; final weights and standardization are fitted on train only.

Codex-assisted checks and other AI implementation reviews are internal checks,
not independent human peer review or a guarantee of correctness. The project
author remains responsible for understanding and verifying the method, its
claims, its use of data and the final submitted material.

Reused components are the course package's unchanged data format, numeric
parser, prediction/scoring tools and data manifest; NumPy provides array
operations, and SciPy provides L-BFGS-B optimization during training only.
Questions and references derive from GSM8K under the retained MIT license.
Candidate solutions and representations were supplied by the course using
Qwen3-0.6B, as recorded in the official data manifest. No Qwen weights are
distributed or loaded by this method. Retain `code/data/LICENSE.txt` and
`code/data/NOTICE.txt` with the package.
