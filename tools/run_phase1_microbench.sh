#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

PHASE=phase1 bash "$(dirname "${BASH_SOURCE[0]}")/run_phase_profile.sh" "$@"
