#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ADDON_PARENT_DIR="$REPO_ROOT/2d-platformer-demo/addons"
ADDON_NAME="artsyn"
ADDON_DIR="${ARTSYN_ADDON_DIR:-$ADDON_PARENT_DIR/$ADDON_NAME}"
DIST_DIR="${ARTSYN_DIST_DIR:-$REPO_ROOT/dist}"
PACKAGE_MODE="${ARTSYN_PACKAGE_MODE:-full}"
PACKAGE_NAME="${ARTSYN_PACKAGE_NAME:-}"
DRY_RUN="${ARTSYN_DRY_RUN:-0}"
BIN_DIR="$ADDON_DIR/bin"
BIN_CONTENTS_DIR="$BIN_DIR"
EXTENSION_STEM="${ARTSYN_EXTENSION_STEM:-libmy_extension}"
RUNTIME_PROFILE="${ARTSYN_PACKAGE_RUNTIME_PROFILE:-}"
DEBUG_BINARY="$BIN_DIR/$EXTENSION_STEM.debug.so"
RELEASE_BINARY="$BIN_DIR/$EXTENSION_STEM.release.so"

if [[ ! -d "$ADDON_DIR" ]]; then
    echo "Addon directory not found: $ADDON_DIR" >&2
    exit 1
fi

debug_present=false
release_present=false

if [[ -f "$DEBUG_BINARY" ]]; then
    debug_present=true
fi

if [[ -f "$RELEASE_BINARY" ]]; then
    release_present=true
fi

if [[ "$PACKAGE_MODE" != "core" ]] && [[ ! -d "$BIN_DIR" ]]; then
    echo "Runtime directory not found: $BIN_DIR" >&2
    exit 1
fi

if [[ "$PACKAGE_MODE" != "core" ]]; then
    BIN_CONTENTS_DIR="$(cd "$BIN_DIR" && pwd -P)"
fi

if [[ "$PACKAGE_MODE" != "core" ]] && [[ "$debug_present" == false ]] && [[ "$release_present" == false ]]; then
    echo "No extension binaries found in $BIN_DIR" >&2
    echo "Expected at least one of:" >&2
    echo "  $DEBUG_BINARY" >&2
    echo "  $RELEASE_BINARY" >&2
    exit 1
fi

detect_runtime_profile() {
    if find "$BIN_CONTENTS_DIR" -maxdepth 1 \( -type f -o -type l \) -name 'libonnxruntime*.so*' | grep -q .; then
        local onnx_profile="onnxruntime"
        if find "$BIN_CONTENTS_DIR" -maxdepth 1 \( -type f -o -type l \) -name 'libonnxruntime_providers_cuda.so*' | grep -q .; then
            onnx_profile="onnxruntime-cuda"
        fi

        if find "$BIN_CONTENTS_DIR" -maxdepth 1 \( -type f -o -type l \) \( -name 'libtorch*.so*' -o -name 'libc10*.so*' \) | grep -q .; then
            echo "hybrid"
            return
        fi

        echo "$onnx_profile"
        return
    fi

    if find "$BIN_CONTENTS_DIR" -maxdepth 1 \( -type f -o -type l \) \( -name 'libtorch*.so*' -o -name 'libc10*.so*' \) | grep -q .; then
        echo "nvidia-libtorch"
        return
    fi

    echo "native"
}

if [[ -z "$RUNTIME_PROFILE" ]] && [[ "$PACKAGE_MODE" != "core" ]]; then
    RUNTIME_PROFILE="$(detect_runtime_profile)"
fi

variant_suffix=""
if [[ "$debug_present" == true ]] && [[ "$release_present" == true ]]; then
    variant_suffix=""
elif [[ "$debug_present" == true ]]; then
    variant_suffix="-debug"
elif [[ "$release_present" == true ]]; then
    variant_suffix="-release"
fi

case "$PACKAGE_MODE" in
    full)
        PACKAGE_NAME="${PACKAGE_NAME:-artsyn-linux-x86_64-${RUNTIME_PROFILE}${variant_suffix}}"
        ;;
    core)
        PACKAGE_NAME="${PACKAGE_NAME:-artsyn-core}"
        ;;
    runtime)
        PACKAGE_NAME="${PACKAGE_NAME:-artsyn-linux-x86_64-${RUNTIME_PROFILE}${variant_suffix}-runtime}"
        ;;
    *)
        echo "Unsupported ARTSYN_PACKAGE_MODE: $PACKAGE_MODE" >&2
        echo "Expected one of: full, core, runtime" >&2
        exit 1
        ;;
esac

PACKAGE_PATH="$DIST_DIR/$PACKAGE_NAME.zip"

list_common_files() {
    printf '%s\n' \
        "addons/$ADDON_NAME/ArtSyn_extension.gdextension" \
        "addons/$ADDON_NAME/plugin.cfg" \
        "addons/$ADDON_NAME/plugin.gd" \
        "addons/$ADDON_NAME/README.md"

    if [[ -d "$ADDON_DIR/models" ]]; then
        (
            cd "$ADDON_DIR/models"
            find . -type f | sed "s#^\./#addons/$ADDON_NAME/models/#" | sort
        )
    fi
}

list_runtime_files() {
    (
        cd "$BIN_CONTENTS_DIR"
        find . -maxdepth 1 \( -type f -o -type l \) -name '*.so*' | sed "s#^\./#addons/$ADDON_NAME/bin/#" | sort
    )
}

list_runtime_metadata_files() {
    printf '%s\n' "addons/$ADDON_NAME/README.md"
}

if [[ "$DRY_RUN" == "1" ]]; then
    echo "Dry run package mode: $PACKAGE_MODE"
    if [[ "$PACKAGE_MODE" != "core" ]]; then
        echo "Runtime profile: $RUNTIME_PROFILE"
    fi
    echo "Resolved package path: $PACKAGE_PATH"
    if [[ "$debug_present" == true ]]; then
        echo "Including debug binary: $(basename "$DEBUG_BINARY")"
    fi
    if [[ "$release_present" == true ]]; then
        echo "Including release binary: $(basename "$RELEASE_BINARY")"
    fi
    echo "Staged files:"
    case "$PACKAGE_MODE" in
        full)
            list_common_files
            list_runtime_files
            ;;
        core)
            list_common_files
            ;;
        runtime)
            list_runtime_metadata_files
            list_runtime_files
            ;;
    esac
    exit 0
fi

STAGING_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGING_DIR"' EXIT

mkdir -p "$DIST_DIR" "$STAGING_DIR/addons/$ADDON_NAME"
rm -f "$PACKAGE_PATH"

copy_common_files() {
    cp "$ADDON_DIR/ArtSyn_extension.gdextension" "$STAGING_DIR/addons/$ADDON_NAME/"
    cp "$ADDON_DIR/plugin.cfg" "$STAGING_DIR/addons/$ADDON_NAME/"
    cp "$ADDON_DIR/plugin.gd" "$STAGING_DIR/addons/$ADDON_NAME/"
    cp "$ADDON_DIR/README.md" "$STAGING_DIR/addons/$ADDON_NAME/"

    mkdir -p "$STAGING_DIR/addons/$ADDON_NAME/models"
    cp -R "$ADDON_DIR/models/." "$STAGING_DIR/addons/$ADDON_NAME/models/"
}

copy_runtime_files() {
    cp "$ADDON_DIR/README.md" "$STAGING_DIR/addons/$ADDON_NAME/"
    mkdir -p "$STAGING_DIR/addons/$ADDON_NAME/bin"
    find "$BIN_CONTENTS_DIR" -maxdepth 1 -type f -name '*.so*' -exec cp '{}' "$STAGING_DIR/addons/$ADDON_NAME/bin/" ';'
    find "$BIN_CONTENTS_DIR" -maxdepth 1 -type l -name '*.so*' -exec cp -a '{}' "$STAGING_DIR/addons/$ADDON_NAME/bin/" ';'
}

case "$PACKAGE_MODE" in
    full)
        copy_common_files
        copy_runtime_files
        ;;
    core)
        copy_common_files
        ;;
    runtime)
        copy_runtime_files
        ;;
esac

(
    cd "$STAGING_DIR"
    cmake -E tar cf "$PACKAGE_PATH" --format=zip addons
)

echo "Created addon package: $PACKAGE_PATH"
