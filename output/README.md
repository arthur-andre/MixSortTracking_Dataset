# Output

Per-frame tracking overlays (`frame1.jpg`, `frame2.jpg`, …) from `yolox/utils/visualize.py` when visualization is enabled.

**README demo:** stitch to GIF:

```shell
bash scripts/make_demo_gif.sh 12
# → assets/demo.gif
```

Or convert your MP4: `ffmpeg -i output.mp4 -vf "fps=12,scale=1280:-1" -loop 0 ../assets/demo.gif`

Gitignored except this file.
