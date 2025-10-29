# AR Tracker & NLP/Transformers Exercises

This repository contains two main projects:

1. **Augmented Reality Object Tracking** using AKAZE and OpenCV (C++)
2. **NLP/Transformers Exercises** (Python) covering self-attention, sentiment analysis, classification, text generation, summarization, translation, and more.

---

## Part 1: AR Tracking with AKAZE (C++)

### Overview

The AR tracker detects a reference object in a video and overlays its bounding polygon in real-time using AKAZE features and homography.

**Workflow:**
1. Load reference image and compute AKAZE keypoints/descriptors.
2. Open video file for processing.
3. Detect keypoints in each frame.
4. Match descriptors using:
   - KNN matching + Lowe's ratio test
   - Symmetric matches for bijective filtering
5. Compute homography if enough good matches exist.
6. Overlay the object's corners on the frame.
7. Display info: number of matches, mode (Detection/Tracking), processing time.
8. User controls:
   - **Space**: Pause/Play
   - **Escape**: Quit
   - **Trackbar**: Seek frames

### Requirements

- OpenCV 4.x
- C++17 compiler (g++)
- Linux or Windows

### Compilation

```bash
g++ -O2 -std=c++17 -o ar_tracker ar_tracker.cpp `pkg-config --cflags --libs opencv4`
