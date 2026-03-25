#!/usr/bin/env bash

# shellcheck shell=bash

load_pretraining_repo_env() {
  local repo_root="$1"
  local env_file
  local env_files=()

  if [[ -n "${PRETRAINING_ENV_FILE:-}" ]]; then
    env_files+=("${PRETRAINING_ENV_FILE}")
  fi

  env_files+=(
    "${repo_root}/config/runtime/pretraining.env"
    "${repo_root}/config/runtime/pretraining.env.local"
    "${repo_root}/config/runtime/hf_upload.env"
    "${repo_root}/config/runtime/hf_upload.env.local"
    "${repo_root}/.env.pretraining"
    "${repo_root}/.env.pretraining.local"
    "${repo_root}/.env.hf_upload"
    "${repo_root}/.env.hf_upload.local"
  )

  for env_file in "${env_files[@]}"; do
    if [[ -f "${env_file}" ]]; then
      set -a
      # shellcheck disable=SC1090
      source "${env_file}"
      set +a
    fi
  done
}
