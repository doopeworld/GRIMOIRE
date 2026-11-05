#!/bin/bash
# 11-05: the regression's mu-s case (Muse-Glimmer-30B INT4-W4A16, plain decode, short prompt) changed wording today
# after three identical runs (10-30, 11-01, 11-02), although no INT4 / Muse code changed. Twice with the v1.10.3 CLI
# (grimoire-engine-1101/bin, the 11-01 build) and twice with the candidate's (grimoire-release-1105/bin), the
# regression's own command line. Holds logs/HOLD-mus (the B580 queue waits); starts after the gate ends.
set -u
R=/mnt/storage/isos/grimoire-runs
B=$R/bench-1022
exec >"$B/logs/mus-check-1105.out" 2>&1
touch "$B/logs/HOLD-mus"
trap 'rm -f "$B/logs/HOLD-mus"' EXIT
gate_over() { awk '/^V1104B_ALL_CONTINUE/{c=1; e=0} c && /^V1104B_ALL_(DONE|STOP)/{e=1} END{exit !(c && e)}' "$B/logs/v1104b-all.out"; }
until gate_over; do sleep 30; done
until [ -z "$(docker ps -q)" ]; do sleep 10; done
exec 9>"$R/gpu0.lock"
flock -n 9 || { echo "gpu0.lock held"; exit 75; }
cd /mnt/storage/isos/grimoire-fuse || exit 9
SHORT="Explain in two sentences why the sky is blue."
h() { awk '/norms /{f=1;next} f' "$1" | grep -vE '^\s|^$|prompt=|prompt ids|greedy ids' | md5sum | cut -c1-12; }
# third variant: the candidate's executables with v1.10.3's libgrimoire_gemm.so (Muse's INT4 prompt GEMMs live there)
X=/mnt/storage/isos/grimoire-runs/mus-tree-1105
rm -rf "$X"; mkdir -p "$X/bin"
cp -P /mnt/storage/isos/grimoire-release-1105/bin/* "$X/bin/"
cp /mnt/storage/isos/grimoire-release-1101/bin/libgrimoire_gemm.so "$X/bin/libgrimoire_gemm.so"
cp -a /mnt/storage/isos/grimoire-release-1105/tools "$X/" 2>/dev/null
for spec in "v1103a /mnt/storage/isos/grimoire-engine-1101" "v1103b /mnt/storage/isos/grimoire-engine-1101" \
            "candA /mnt/storage/isos/grimoire-release-1105" "candB /mnt/storage/isos/grimoire-release-1105" \
            "candOldLib $X"; do
  set -- $spec
  GRIM_TREE=$2 EXTRA_ENV="" LIM=900 bash tools/g0run.sh "mus-$1" -m /models/Muse-Glimmer-30B-INT4-W4A16 --proj int4 \
    --ctx 8192 -p "$SHORT" -n 64 9>&- > /dev/null
  cp "/tmp/grim-mus-$1.log" "$B/logs/mus-$1-1105.log"
  echo "$1 $(h "$B/logs/mus-$1-1105.log")"
done
echo "regress 11-02: $(h $R/regress-1102-1701/mu-s.log)  regress today: $(h $R/regress-1105-1154/mu-s.log)"
echo "MUS_CHECK_DONE $(date +%T)"
