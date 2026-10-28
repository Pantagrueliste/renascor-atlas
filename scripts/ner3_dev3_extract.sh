#!/bin/bash
# dev3: extract the archive tarballs needed by ner3_dev3_build.py into ROOT (same layout as the gold v2 build). The archive itself is never modified.
#   TARS=<dir with the tar.gz> ROOT=<dir> bash scripts/ner3_dev3_extract.sh
set -euo pipefail
: "${TARS:?}"; : "${ROOT:?}"
mkdir -p "$ROOT"/{estilometria,galileo,ws_it,liberliber,raw__LatinWikisource,raw__SpanishGoldenAgeProse_WS,raw__WikisourceES_Renaissance,noscemus}
x() { tar xzf "$TARS/$1" -C "$ROOT/$2"; }
x raw__estilometria.tar.gz estilometria
x raw__Galileo_Opere_Favaro.tar.gz galileo
x nas__ws_it_renaissance.tar.gz ws_it
x raw__liberliber.tar.gz liberliber
x raw__LatinWikisource.tar.gz raw__LatinWikisource
x raw__SpanishGoldenAgeProse_WS.tar.gz raw__SpanishGoldenAgeProse_WS
x raw__WikisourceES_Renaissance.tar.gz raw__WikisourceES_Renaissance
tar xzf "$TARS/raw__bibit_tei.tar.gz" -C "$ROOT" && rm -rf "$ROOT/bibit" && mv "$ROOT/bibit_tei" "$ROOT/bibit"
tar xzf "$TARS/raw__latineebo.tar.gz" -C "$ROOT"
tar xzf "$TARS/raw__salamanca.tar.gz" -C "$ROOT"
tar xzf "$TARS/raw__impactes.tar.gz" -C "$ROOT"
tar xzf "$TARS/nas__ws_es_renaissance.tar.gz" -C "$ROOT"
tar xzf "$TARS/raw__noscemus.tar.gz" -C "$ROOT/noscemus"   # the zip stays packed; the builder reads it member by member
find "$ROOT/noscemus" -name '*.zip' | head -1
echo extracted
