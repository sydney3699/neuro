#!/usr/bin/env bash
# Build (and optionally push) the per-stage images. Run from the repo root:
#   dockerfiles/build.sh                       # build all as neuro/neuro-<stage>:latest
#   REGISTRY=123456789.dkr.ecr.us-east-1.amazonaws.com TAG=v1 dockerfiles/build.sh
#   PUSH=1 REGISTRY=<ecr-uri> dockerfiles/build.sh   # build + push (for the awsbatch profile)
set -euo pipefail

REGISTRY="${REGISTRY:-neuro}"
TAG="${TAG:-latest}"
PUSH="${PUSH:-0}"

# stage -> Dockerfile
STAGES=(envi scvi scgpt stagate analysis banksy)

cd "$(dirname "$0")/.."   # repo root = build context

for stage in "${STAGES[@]}"; do
    image="${REGISTRY}/neuro-${stage}:${TAG}"
    echo ">>> building ${image}"
    docker build -f "dockerfiles/Dockerfile.${stage}" -t "${image}" .
    if [[ "${PUSH}" == "1" ]]; then
        echo ">>> pushing ${image}"
        docker push "${image}"
    fi
done

# For AWS Batch: authenticate to ECR once before PUSH=1, e.g.
#   aws ecr get-login-password --region "$AWS_REGION" \
#     | docker login --username AWS --password-stdin "$REGISTRY"
# and ensure each repo exists:  aws ecr create-repository --repository-name neuro-<stage>
echo "done."
