# Deploy the model API on Vercel

## What changed
- random_forest.joblib: 179 MB -> 28.8 MB (lzma level 6). xgboost.joblib: 1.2 MB -> 0.3 MB.
  Predictions are bit-for-bit identical to the originals (checked on a sample car).
- app.py -> api/index.py. Models now load lazily (only the one requested), which keeps memory and cold start low.
- requirements.txt uses `xgboost-cpu`. Plain `xgboost` pulls ~345 MB of NVIDIA libraries and breaks Vercel's 500 MB limit.
- Versions are pinned to the ones that load your pickles (scikit-learn 1.5.2). Do not upgrade without retraining.
- Optional: set MODEL_API_KEY in Vercel. The API then requires an `x-api-key` header (your Node backend already sends it).

## Steps
1. Push this folder to a GitHub repo. The .joblib files are under 100 MB, so plain git works.
2. vercel.com > Add New > Project > import the repo. Framework: Other. Leave build settings empty.
3. (Optional) Settings > Environment Variables: MODEL_API_KEY. Set the same value on your Node backend.
4. Deploy. Test: https://<project>.vercel.app/health, then POST /predict.
5. On your Node backend set MODEL_API_URL=https://<project>.vercel.app (no trailing slash).

CLI alternative: `npm i -g vercel && vercel --prod`.

## If the deploy fails on size
Vercel allows 500 MB unzipped for Python functions. This bundle is close to that. Options, in order:
1. Deploy only one model: delete the other from models/ and from FILES in api/index.py.
2. Keep models out of the bundle: upload them to Vercel Blob (or any storage), download to /tmp on first request, then joblib.load from there.
3. Use Render, Railway or Hugging Face Spaces (Docker) with the original app.py and gunicorn. They have no 500 MB limit and no 60 s timeout.

## Notes
- First request after idle takes a few seconds (loading + decompressing the forest, about 2.5 s plus Python start). Your backend's 30 s timeout is enough.
- Free (Hobby) plan limits requests to 60 s max duration; vercel.json sets that.
- Keep the lzma level at 6. Level 9 saves ~7 MB more but takes several minutes to write.
