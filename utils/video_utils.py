import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        frames.append(frame)

    cap.release()
    return frames


def save_video(output_video_frames, output_path, fps = 25):
    # we don't know the fps of the original video as we dont have cap here so hardcoding it to 25 fps which is quite standard
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = output_video_frames[0].shape[:2]
    frame_size = (width, height)

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        frame_size
    )

    for frame in output_video_frames:
        writer.write(frame)

    writer.release()