from ultralytics import YOLO

model = YOLO("/Users/aakash/football_larp/models/best.pt") #load a pretrained model


results = model.predict("/Users/aakash/football_larp/input_videos/08fd33_4.mp4")


frames_with_balls = 0
total_frames = len(results)
for r in results:
    class_ids = r.boxes.cls.tolist()
    if 0.0 in class_ids:
        frames_with_balls += 1

print(f"times balls detected: {frames_with_balls} of {total_frames} frames")