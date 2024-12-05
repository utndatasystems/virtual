import sys
import os

DEBUG_MODE = False

def _block_print():
  sys.stdout = open(os.devnull, 'w')

def _enable_print():
  sys.stdout = sys.__stdout__

def _check_debug_mode():
  if DEBUG_MODE:
    _enable_print()
  else:
    _block_print()

