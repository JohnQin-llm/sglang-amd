"""Run argv[1:] as one command (for k2safe.sh, which only runs python scripts). Env passes through."""
import os, sys
os.execvp(sys.argv[1], sys.argv[1:])
