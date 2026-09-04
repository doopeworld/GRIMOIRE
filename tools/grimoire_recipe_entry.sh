#!/bin/sh
# Applies the per-model recipe in /opt/grimoire/recipes, then execs the real
# binary.  Installed AT the binary's own path (the real one is <name>.real) so
# it still runs when a container overrides --entrypoint, as the Unraid
# templates do.
#
# Rules, in order of precedence:
#   1. A variable already set in the environment ALWAYS wins and is reported
#      as an override.  The container can still experiment.
#   2. Otherwise the recipe value is applied.
#   3. No recipe file for this model  ->  nothing is changed at all.
# Rule 3 is why this is safe for models that have no recipe (Muse, Ornith):
# the wrapper is inert and simply execs.
set -u

SELF="$0"
REAL="${SELF}.real"
[ -x "$REAL" ] || { echo "grimoire-entry: missing $REAL" >&2; exit 127; }

# Find the model path: --model <p>, --model=<p>, -m <p>.
MODEL=""
prev=""
for a in "$@"; do
    case "$prev" in --model|-m) MODEL="$a" ;; esac
    case "$a" in --model=*) MODEL="${a#--model=}" ;; esac
    prev="$a"
done

if [ -n "$MODEL" ]; then
    NAME=$(basename "$MODEL")
    RECIPE="/opt/grimoire/recipes/${NAME}.env"
    if [ -f "$RECIPE" ]; then
        applied=""; overridden=""
        while IFS= read -r line; do
            case "$line" in ''|\#*) continue ;; esac
            key=${line%%=*}
            val=${line#*=}
            [ "$key" = "$line" ] && continue
            cur=$(printenv "$key" 2>/dev/null) || cur=""
            if [ -n "$cur" ]; then
                [ "$cur" = "$val" ] || overridden="$overridden $key=$cur(recipe:$val)"
            else
                export "$key=$val"
                applied="$applied $key=$val"
            fi
        done < "$RECIPE"
        echo "  recipe       ${NAME}.env"
        [ -n "$applied" ]    && echo "  recipe applied  :$applied"
        [ -n "$overridden" ] && echo "  recipe OVERRIDDEN:$overridden  <-- container env wins"
    else
        echo "  recipe       none for ${NAME} (no flags applied)"
    fi
fi

exec "$REAL" "$@"
