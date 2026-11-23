#!/bin/bash
# usage: fetch_month.sh topic year month outdir ; parallel resumable range download with size check
T=$1; Y=$2; M=$3; OUT=$4; P=${P:-24}
URL="https://opendata.manage-roadworks.service.gov.uk/$T/$Y/$M.zip"
mkdir -p "$OUT"; cd "$OUT" || exit 1
SIZE=$(curl -sI -m 60 "$URL" | grep -i '^content-length' | tr -dc '0-9')
[ -z "$SIZE" ] && { echo "FAIL nosize $URL"; exit 1; }
if [ -f "$M.zip" ] && [ "$(stat -c %s "$M.zip")" = "$SIZE" ]; then echo "SKIP $T/$Y/$M"; exit 0; fi
# parts kept for resume (same P)
CH=$(( (SIZE + P - 1) / P )); pids=()
for i in $(seq 0 $((P-1))); do
  S=$((i*CH)); E=$((S+CH-1)); [ $E -ge $SIZE ] && E=$((SIZE-1)); [ $S -gt $E ] && continue
  ( W=$((E-S+1)); for a in $(seq 1 30); do
      HAVE=0; [ -f "$M.part$i" ] && HAVE=$(stat -c %s "$M.part$i")
      [ "$HAVE" -ge "$W" ] && break
      curl -s -m 3600 -r $((S+HAVE))-$E "$URL" >> "$M.part$i"; sleep 2
    done ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
cat $(for i in $(seq 0 $((P-1))); do [ -f "$M.part$i" ] && echo "$M.part$i"; done) > "$M.zip.tmp"
if [ "$(stat -c %s "$M.zip.tmp")" = "$SIZE" ]; then mv "$M.zip.tmp" "$M.zip"; rm -f $M.part*; echo "DONE $T/$Y/$M $SIZE"; else echo "FAIL size $T/$Y/$M"; rm -f "$M.zip.tmp" $M.part*; exit 1; fi
