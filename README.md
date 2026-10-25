# TraceX – ANPR vehicle trajectory tracing (SIH26127 / Hacktober Hack Days 2026)

AI worker (YOLO11 + ByteTrack + OCR) -> Cloudinary (authenticated storage + metadata + face blur on delivery) -> FastAPI + DB -> live map dashboard.

## Run locally
```bash
cp .env.example .env            # add your CLOUDINARY_URL
cd backend && pip install -r requirements.txt
export $(grep -v '^#' ../.env | xargs) && uvicorn main:app --reload
# open http://localhost:8000 and click "Load demo data"
```
Worker (second terminal, a GPU is not required for yolo11n):
```bash
cd worker && pip install -r requirements.txt
export $(grep -v '^#' ../.env | xargs)
python worker.py --source traffic.mp4 --camera-id cam01 --name "MG Road" --lat 12.9756 --lng 77.6066
python worker.py --source traffic2.mp4 --camera-id cam02 --name "Trinity Circle" --lat 12.9730 --lng 77.6200
```
Run it once per "camera" with different clips/coordinates, then search the plate in the dashboard.

## Deploy
1. Cloudinary: create a free account, copy the API environment variable (`CLOUDINARY_URL`) from the dashboard.
2. Database: create a free Supabase (or Neon) project and copy its Postgres connection string as `DATABASE_URL`.
3. GitHub: push this folder.
4. Render: New > Web Service > connect repo > Root Directory `backend` > Runtime Docker. Environment variables: `CLOUDINARY_URL`, `DATABASE_URL`, `WORKER_KEY` (long random string), optional `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.
5. Open your Render URL, click "Load demo data" to check it works.
6. Run the worker on your laptop pointing at the live API: `export API_URL=https://YOUR-APP.onrender.com WORKER_KEY=... CLOUDINARY_URL=...` then the `python worker.py ...` commands above.

## Better plate accuracy
Download an Indian number-plate dataset (Roboflow Universe/Kaggle, YOLO format), train in Colab with `yolo train model=yolo11n.pt data=data.yaml epochs=50`, and save the result as `worker/plate.pt`. The worker uses it automatically.
