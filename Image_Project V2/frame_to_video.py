import os
import cv2

def create_video_from_frames(frames_folder, output_video, fps=1):
    def numeric_sort(filename):
        name, _ = os.path.splitext(filename)
        number = ''.join(filter(str.isdigit, name))
        return int(number) if number else 0

    frame_files = sorted(
        [f for f in os.listdir(frames_folder) if f.lower().endswith(('.jpg', '.png'))],
        key=numeric_sort
    )

    if not frame_files:
        print("Hiç görüntü bulunamadı.")
        return

    first_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i, frame_file in enumerate(frame_files):
        img_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(img_path)
        if frame is not None:
            out.write(frame)
            print(f"{i+1}/{len(frame_files)} eklendi: {frame_file}")
        else:
            print("Hatalı frame:", frame_file)

    out.release()
    print("🎉 Video oluşturuldu:", output_video)

if __name__ == "__main__":
    create_video_from_frames(
        frames_folder=r'C:\Users\kralm\OneDrive\Belgeler\frames',
        output_video=r'C:\Users\kralm\OneDrive\Belgeler\frames_output\frames_video.mp4',
        fps=1
    )
