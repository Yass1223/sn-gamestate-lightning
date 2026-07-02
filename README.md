<div align="center">

# SoccerNet Game State Reconstruction — 

**End-to-end athlete tracking & identification on a minimap.**

This repository is the [SoccerNet Game State Reconstruction (GSR)](https://www.soccer-net.org/tasks/new-game-state-reconstruction) baseline (built on [TrackLab](https://github.com/TrackingLaboratory/tracklab)),

![Game State Reconstruction example](images/gamestate-example.jpg)

</div>

---

## Tracker: BoT-SORT · SOF + GTA-Link (default)

The default tracker (`soccernet_botsort` config) replaces the baseline StrongSORT with the
winning combination from our BoT-SORT camera-motion-compensation ablation
(`yolov11-bot-sort-cmc.ipynb`):

- **Detector** — YOLO11-L fine-tuned on SoccerNet (single-class person), fetched from Hugging Face.
- **Tracker** — BoT-SORT with **SOF** (sparse optical flow) CMC and a sports-OSNet ReID
  (`sn_gamestate/configs/modules/track/botsort_osnet.yaml`).
- **GTA-Link** — offline tracklet stitching (agglomerative clustering on EMA-averaged OSNet
  embeddings with temporal/spatial gating), run right after `track`
  (`sn_gamestate/track/gta_link_api.py`), with hyperparameters tuned on the valid split.

Ablation on the GSR **test** split (49 clips, YOLO11L detections, tracker-only HOTA):

| CMC method | raw HOTA | + GTA-Link |
|---|---|---|
| **SOF** | **0.6687** | **0.6947** |
| ECC | 0.6643 | 0.6924 |
| ORB | 0.6511 | 0.6809 |
| SIFT | 0.6315 | 0.6579 |
| none (CMC off) | 0.6315 | 0.6579 |

Run locally with `tracklab -cn soccernet_botsort` (the original baseline remains available
via `tracklab -cn soccernet`).

## Running on Lightning AI

The GitHub Action **"SoccerNet GSR Evaluation on Lightning AI"** (manual trigger) dispatches
the evaluation to a Lightning Studio GPU job. Pick the split(s), number of videos, machine
(e.g. **L4**), and the tracker config (`soccernet_botsort` by default). The same runner can
be used directly inside a Studio:

```bash
git clone https://github.com/Yass1223/sn-gamestate-lightning.git
cd sn-gamestate-lightning
SPLITS="test" NVID=-1 CONFIG_NAME=soccernet_botsort bash scripts/lightning_eval.sh
```
