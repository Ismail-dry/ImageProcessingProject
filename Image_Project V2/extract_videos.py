import cv2
import os

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Hata: Video açılamadı!")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = int(total_frames / fps)

    print(f"Video işleniyor: {video_path}")
    print(f"Video FPS: {fps}")
    print(f"Toplam süre (saniye): {duration_seconds}")
    print(f"Toplam frame sayısı: {total_frames}")

    for sec in range(duration_seconds):
        frame_number = int(sec * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = video.read()

        if not success:
            print(f"{sec}. saniyedeki frame okunamadı.")
            continue

        output_path = os.path.join(output_folder, f"frame_{sec:06d}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"{sec}. saniyede frame kaydedildi: {output_path}")

    video.release()
    print("İşlem tamamlandı!")

# Kullanım
if __name__ == "__main__":
    video_path = r"C:\Users\kralm\Downloads\Segment_024.avi"
    output_folder = r"C:\Users\kralm\OneDrive\Belgeler\frames"
    extract_frames(video_path, output_folder)
