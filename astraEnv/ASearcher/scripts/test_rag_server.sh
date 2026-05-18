#!/bin/bash
set -euo pipefail

# Test RAG server endpoints.
#
# Usage:
#   bash scripts/test_rag_server.sh [server_address]
#
# If no address is given, it reads from tmp-log/rag_server_addrs/.

if [[ $# -ge 1 ]]; then
  ADDR="$1"
else
  ADDR_DIR="./tmp-log/rag_server_addrs"
  ADDR_FILE=$(ls "${ADDR_DIR}"/*.txt 2>/dev/null | head -1)
  if [[ -z "${ADDR_FILE}" ]]; then
    echo "No server address found in ${ADDR_DIR}. Pass address as argument."
    exit 1
  fi
  ADDR=$(cat "${ADDR_FILE}")
  echo "Found server address: ${ADDR}"
fi

BASE_URL="http://${ADDR}"
PASS=0
FAIL=0

echo ""
echo "=== Test 1: /retrieve endpoint ==="
RESP=$(curl -s --max-time 10 -X POST "${BASE_URL}/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"queries": ["What is the capital of France?"], "topk": 3, "return_scores": true}')

if echo "${RESP}" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d['result'][0])==3" 2>/dev/null; then
  echo "PASSED - Retrieved 3 documents with scores."
  PASS=$((PASS+1))
else
  echo "FAILED - Unexpected response:"
  echo "${RESP}"
  FAIL=$((FAIL+1))
fi

echo ""
echo "=== Test 2: /retrieve batch queries ==="
RESP=$(curl -s --max-time 10 -X POST "${BASE_URL}/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"queries": ["What is Python?", "Who invented the telephone?"], "topk": 2}')

if echo "${RESP}" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d['result'])==2 and len(d['result'][0])==2" 2>/dev/null; then
  echo "PASSED - Batch retrieval returned 2 results with 2 docs each."
  PASS=$((PASS+1))
else
  echo "FAILED - Unexpected response:"
  echo "${RESP}"
  FAIL=$((FAIL+1))
fi

echo ""
echo "=== Test 3: /access endpoint ==="
# Use a URL from the retrieve result
URL=$(echo "${RESP}" | python3 -c "import sys,json; print(json.load(sys.stdin)['result'][0][0]['url'])" 2>/dev/null || echo "")

if [[ -n "${URL}" ]]; then
  RESP=$(curl -s --max-time 10 -X POST "${BASE_URL}/access" \
    -H "Content-Type: application/json" \
    -d "{\"urls\": [\"${URL}\"]}")

  if echo "${RESP}" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['result'][0] is not None" 2>/dev/null; then
    echo "PASSED - Page access returned content."
    PASS=$((PASS+1))
  else
    echo "FAILED - Unexpected response:"
    echo "${RESP}"
    FAIL=$((FAIL+1))
  fi
else
  echo "SKIPPED - Could not extract URL from previous test."
  FAIL=$((FAIL+1))
fi

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
if [[ ${FAIL} -gt 0 ]]; then
  exit 1
fi
