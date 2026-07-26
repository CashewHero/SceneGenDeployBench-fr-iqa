# Runner Wrapper

`runner_wrapper/` turns a model repository into a SceneGenDeployBench runner image. It provides the HTTP server, job logging, resource measurements, Docker wiring, examples, and local test helper. A model repository normally only needs a model-specific `adapter.py` and runner catalog.

The directory `runner_wrapper/` is self-contained so it can be copied or pulled as a subtree without the main repository.

One runner catalog entry has one role:

- A generator turns dataset inputs into reusable generated files.
- An evaluator consumes dataset data and/or files from a generator and reports metrics.

## Add To A Model Repository

From the model repository root:

```bash
git remote add deploybench https://github.com/CashewHero/SceneGenDeployBench.git
git fetch deploybench subtree/runner_wrapper
git subtree add --prefix=runner_wrapper deploybench subtree/runner_wrapper --squash
```

Pull later updates with:

```bash
git fetch deploybench subtree/runner_wrapper
git subtree pull --prefix=runner_wrapper deploybench subtree/runner_wrapper --squash
```

The main files are:

```text
runner_wrapper/
  adapter.py       model-specific job implementation
  files.py         compatible artifact publication
  server.py        shared HTTP runner server
  Dockerfile       runner image build
  localtest.sh     local build and smoke helper
  AGENTS.md        detailed adaptation contract
  examples/        request, catalog, Docker, and workflow templates
```

Copy the matching catalog template to `runner_wrapper/config/runners/<runner>.yaml` and edit it for the model. To use the runner locally, copy that catalog into the active DeployBench runner-config directory.

## Build And Test

Build from the model repository root:

```bash
docker build -f runner_wrapper/Dockerfile -t my-model-runner .
```

Or use the helper:

```bash
runner_wrapper/localtest.sh build
runner_wrapper/localtest.sh smoke
```

## Data Flow

The orchestrator supplies the selected dataset data to a runner. An evaluator can also receive generated files and additional dataset viewpoints. Each runner writes its job folder and reports reusable outputs or metrics back to the orchestrator.

The wire contract is defined in [Runner API](docs/api.md).

Manual run:

```bash
docker run --rm -p 58090:58090 \
  -e RUNNER_NAME=fr_iqa \
  -e RUNNER_TYPE=evaluator \
  -e RUNNER_VERSION=0.1.2 \
  -v "$PWD/data:/data" \
  my-model-runner
```

Submit `runner_wrapper/examples/evaluator_job_request.json` to `POST /run-job`, then poll `GET /status`.

## Publish An Image

Create the image workflow from the included template:

```bash
mkdir -p .github/workflows
cp runner_wrapper/examples/github-workflows/build-runner-image.yaml \
  .github/workflows/runner-image.yaml
```

The target repository should be named `SceneGenDeployBench-<model>`. The workflow derives the GHCR image name from the repository name.
