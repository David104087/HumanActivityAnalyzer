"""
01_extract_landmarks.py

Extrae landmarks por frame usando MediaPipe Pose desde una carpeta de videos
y guarda CSVs preprocesados con columnas 'video','frame', 'nx_0'..'nx_32','ny_0'..'ny_32'.

Uso:
 python 01_extract_landmarks.py --videos_dir data/raw_videos --out_dir data/preprocessed

"""
import os
import argparse
import cv2
import pandas as pd
import mediapipe as mp


def extract(videos_dir, out_dir, max_frames=None):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

    os.makedirs(out_dir, exist_ok=True)
    video_files = [os.path.join(videos_dir, f) for f in os.listdir(videos_dir) if f.lower().endswith(('.mp4','.avi','.mov'))]
    if not video_files:
        print('No se encontraron videos en', videos_dir)
        return

    for vf in video_files:
        cap = cv2.VideoCapture(vf)
        basename = os.path.splitext(os.path.basename(vf))[0]
        rows = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            row = {'video': basename, 'frame': frame_idx}
            if res.pose_landmarks:
                for i, lm in enumerate(res.pose_landmarks.landmark):
                    row[f'nx_{i}'] = lm.x
                    row[f'ny_{i}'] = lm.y
            else:
                # rellenar NaNs si no hay detección
                for i in range(33):
                    row[f'nx_{i}'] = float('nan')
                    row[f'ny_{i}'] = float('nan')

            rows.append(row)
            frame_idx += 1

        cap.release()
        out_csv = os.path.join(out_dir, f"{basename}_preprocessed.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f'Guardado {out_csv} con {len(rows)} frames')

    pose.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos_dir', default='data/raw_videos')
    parser.add_argument('--out_dir', default='data/preprocessed')
    parser.add_argument('--max_frames', type=int, default=None)
    args = parser.parse_args()
    extract(args.videos_dir, args.out_dir, max_frames=args.max_frames)


if __name__ == '__main__':
    main()
