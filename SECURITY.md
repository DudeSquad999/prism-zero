# Security boundaries

This is not a sandbox, theorem prover or code-safety validator. Trusted application
code configures schemas, patterns, paths, checkpoints and rules. Generated text is
data, never executable instructions. Isolate hostile schemas/plugins with OS
CPU/memory/network limits; in-process bounds are not complete DoS protection.
Use private account-owned dataset directories and one writer. Prevent path or
parent-directory replacement by untrusted users. Review models, dependencies,
licenses and training data independently. Do not publish secrets or working
exploits in public issues. No private disclosure channel or response SLA is defined.
