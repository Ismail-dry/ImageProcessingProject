import cv2


def combine_videos_side_by_side(video1_path, video2_path, output_path="combined_output.mp4"):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("❌ Videolardan biri açılamadı.")
        return

    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    fps = min(fps1, fps2)

    frame_count = int(min(cap1.get(cv2.CAP_PROP_FRAME_COUNT), cap2.get(cv2.CAP_PROP_FRAME_COUNT)))

    width = int(min(cap1.get(cv2.CAP_PROP_FRAME_WIDTH), cap2.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(min(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT), cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Codec değişikliği - AVI dosyaları için daha uyumlu
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID codec AVI ile daha uyumlu
    
    try:
        # Önce XVID dene
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        # Test et - başarısız olursa exception fırlatır
        test_frame = cv2.imread(video1_path, cv2.IMREAD_COLOR)
        if test_frame is None:
            raise Exception("Test başarısız")
    except:
        # XVID başarısız olursa mp4v dene
        print("⚠️ XVID codec ile sorun yaşandı, mp4v deneniyor...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    for i in range(frame_count):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1:
            print(f"⚠️ {i+1}. kare input videodan alınamadı.")
            break
        if not ret2:
            print(f"⚠️ {i+1}. kare output videodan alınamadı.")
            break

        frame1 = cv2.resize(frame1, (width, height))
        frame2 = cv2.resize(frame2, (width, height))

        # Renk kontrolü
        if len(frame1.shape) != 3:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
        if len(frame2.shape) != 3:
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

        combined = cv2.hconcat([frame1, frame2])
        out.write(combined)

        print(f"{i+1}/{frame_count} kare işlendi.")

    cap1.release()
    cap2.release()
    out.release()
    print(f"✅ Bitti! Video kaydedildi: {output_path}")





if __name__ == "__main__":
    combine_videos_side_by_side(
        video1_path=r"C:\Users\kralm\OneDrive\Belgeler\frames_output\frames_video.mp4",                   
        video2_path=r"C:\Users\kralm\OneDrive\Belgeler\frames_output\output_video.mp4",      
        output_path=r"C:\Users\kralm\OneDrive\Belgeler\frames_output\combine_output_video.mp4"      
    )