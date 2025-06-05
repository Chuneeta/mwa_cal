import os, sys
try:
    import mwa_cal
except:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../mwa_cal')))
    import mwa_cal