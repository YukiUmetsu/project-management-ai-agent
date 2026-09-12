for f in phase_1/*.py; do
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') START $f ====="

  python "$f"
  status=$?

  echo "===== $(date '+%Y-%m-%d %H:%M:%S') END $f exit=$status ====="
  echo
done 2>&1 | tee "test-run-$(date +%Y%m%d-%H%M%S).log"