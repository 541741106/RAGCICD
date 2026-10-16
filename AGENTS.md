# Project teaching conventions

- Teach in Chinese. Before asking the learner to complete code, explain its goal, inputs, outputs, execution flow, intermediate data shapes, and design reasons.
- Maintain `docs/code-guide.local.md`, a Chinese local learning reference excluded by `.gitignore`. Whenever source files, classes, functions, methods, signatures, behavior, or connections are added or changed, update the reference in the same task after inspecting the actual implementation. For learner-written code, synchronize it when reviewing the saved implementation. If the reference is missing, recreate it from current source.
- Document every project-defined class and function under `src/llmrag`, including standalone functions, and explain external classes and methods actually used by the project. Do not claim to enumerate all internals of third-party packages.
- For each entry describe its file, purpose, inputs/defaults, outputs, execution flow, exceptions, side effects/API costs, callers, and current limitations. Distinguish implemented data paths from planned integrations.
- Keep this learning reference local; never stage it or add secrets to it. Its updates are an assistant workflow convention, not an automated file watcher.
- Preserve learner changes. Explicitly requested implementation can be completed directly; for teaching exercises let the learner implement the key logic.
