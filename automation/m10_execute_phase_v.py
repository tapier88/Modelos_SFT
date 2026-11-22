"""Explicit Phase V entry point using existing verified notebook transport."""
import argparse
import json
import sys
from pathlib import Path
import axxel_colab_start as transport

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--authorize-phase-v',action='store_true',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.authorize_phase_v:raise SystemExit('Explicit training authorization required')
    manifest=json.loads((transport.ROOT/'m06_staging/m10_source_manifest.json').read_text())
    def select(_):
        transport.JOB='phase_v'
        transport.MARKER='# M10_PHASE_V_JOINT_RETURN_V1'
        transport.DIAGNOSTIC=transport.ROOT/'m06_staging/m10_phase_v_payload.py'
        transport.PIN=manifest['payload_sha256']
        transport.RESULT_PREFIX='M10_PHASE_V_RESULT '
        transport.TRAINING_AUTHORIZED=True
        transport.EXECUTION_TIMEOUT_SECONDS=21600
        transport.POLL_SECONDS=20
    transport.select_job=select
    sys.argv=[sys.argv[0],'--output',str(args.output)]
    transport.main()

if __name__=='__main__':main()
