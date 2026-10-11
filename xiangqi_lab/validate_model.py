"""Run the same integration checks used by the website, without a browser."""
import argparse
import json
from engines import check_model, model_catalog, close_workers

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='custom-team-template')
    parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    selected=[m['id'] for m in model_catalog() if m['id']!='human'] if args.all else [args.model]
    passed=True
    try:
        for model in selected:
            try:print(json.dumps(check_model(model),indent=2))
            except Exception as exc:
                passed=False;print(json.dumps({'model':model,'passed':False,'error':str(exc)}))
    finally:close_workers()
    return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
