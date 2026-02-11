# Video to VMD Converter

This project tracks pose motion from a video and exports a basic head-bone VMD animation.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

1. Put your input video in `Input/` (default file is `dance.mp4`).
2. Run:

```bash
python main.py
```

The generated VMD file is saved to `Output/motion_output.vmd`.

## Notes

- If tracking dependencies are missing, the scripts now print clear install instructions.
- `main.py` can fall back to existing `motion_data.npy` when tracking is unavailable.
