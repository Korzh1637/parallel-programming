import argparse
import multiprocessing as mp
import time

import cv2
import numpy as np
from ultralytics import YOLO
from functools import partial


class VideoFile:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self, idx):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        return frame if ret else None

    def __del__(self):
        self.cap.release()


class CameraCapture:
    def __init__(self, camera_id = 0):
        self.cap = cv2.VideoCapture(camera_id)

    def get_frame(self):
        return self.cap.read()

    def __del__(self):
        self.cap.release()

# отрисовка скелета
def draw_keypoints(frame, keypoints_data, conf_threshold = 0.5):
    if not keypoints_data:
        return frame

    # соединяемые пары точек
    skeleton_pairs = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 11), (6, 12)]

    for person in keypoints_data:
        # соединения
        for i1, i2 in skeleton_pairs:
            # проверка на существование
            if i1 < len(person) and i2 < len(person):
                kp1, kp2 = person[i1], person[i2]
                # обе существуют и conf > 0.5
                if kp1 is not None and kp2 is not None and kp1[2] > conf_threshold and kp2[2] > conf_threshold:
                    pt1 = (int(kp1[0]), int(kp1[1]))
                    pt2 = (int(kp2[0]), int(kp2[1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        # точки
        for _, kp in enumerate(person):
            if kp is not None and kp[2] > conf_threshold:
                coord = (int(kp[0]), int(kp[1]))
                cv2.circle(frame, coord, 4, (0, 0, 255), -1)
    return frame

# обработка части видео
def process_chunk(video_path, frame_indices):
    model = YOLO('yolov8s-pose.pt')
    video = VideoFile(video_path)
    results = []

    for idx in frame_indices:
        frame = video.get_frame(idx)
        if frame is None:
            continue

        pose_results = model(frame, verbose=False)
        keypoints_list = []

        for r in pose_results:
            if r.keypoints is not None and len(r.keypoints) > 0:
                # координаты
                kp_data = r.keypoints.xy.cpu().numpy()[0]
                # conf для каждой точки
                kp_conf = r.keypoints.conf.cpu().numpy()[0] if r.keypoints.conf is not None else np.ones(len(kp_data))

                keypoints = np.column_stack((kp_data, kp_conf))
                keypoints_list.append(keypoints)

        if keypoints_list:
            frame = draw_keypoints(frame, keypoints_list)
        results.append((idx, frame))

    return results

# общая функция обработки видео
def process(input_path, output_path, num_workers):
    start = time.time()
    temp = VideoFile(input_path)
    total_frames = temp.total_frames
    fps = temp.fps
    w, h = temp.width, temp.height
    del temp

    # разбиваем индексы кадров на чанки
    indices = list(range(total_frames))
    chunks = np.array_split(indices, num_workers)
    chunks = [chunk.tolist() for chunk in chunks]

    # пул процессов 
    with mp.Pool(processes=num_workers) as pool:
        func = partial(process_chunk, input_path)
        processed_chunks = pool.map(func, chunks)

    all_frames = []
    for chunk in processed_chunks:
        all_frames.extend(chunk)
    all_frames.sort(key=lambda x: x[0])

    # создание выходного видео
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for _, frame in all_frames:
        if frame is not None:
            out.write(frame)
    out.release()

    end = time.time() - start
    print(f"Многопроцессорный режим ({num_workers} процессов): {end:.2f} сек")

# процесс для обработки кадров
def inference_worker(model_path, in_q: mp.Queue, out_q: mp.Queue):
    model = YOLO(model_path)

    while True:
        frame = in_q.get()
        if frame is None:
            break

        results = model(frame, verbose=False)
        keypoints_list = []

        for r in results:
            if r.keypoints is not None and len(r.keypoints) > 0:
                # координаты
                kp_data = r.keypoints.xy.cpu().numpy()[0]
                # conf для каждой точки
                kp_conf = r.keypoints.conf.cpu().numpy()[0] if r.keypoints.conf is not None else np.ones(len(kp_data))

                keypoints = np.column_stack((kp_data, kp_conf))
                keypoints_list.append(keypoints)

        out_q.put(keypoints_list)

# общая функция для вебки
def process_realtime():
    frame_queue = mp.Queue(maxsize=2) # для передачи кадров в процесс инференса
    result_queue = mp.Queue()         # для получения результатов инференса

    infer_proc = mp.Process(target=inference_worker, args=('yolov8s-pose.pt', frame_queue, result_queue))
    infer_proc.start()

    cam = CameraCapture(0)
    fps_counter = 0
    fps_timer = time.time()
    fps_display = 0

    try:
        while True:
            _, frame = cam.get_frame()

            try:
                frame_queue.put_nowait(frame)
            except:
                pass

            keypoints = None
            try:
                keypoints = result_queue.get_nowait()
            except:
                pass

            if keypoints is not None:
                frame = draw_keypoints(frame, keypoints)

            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_timer = time.time()

            cv2.putText(frame, f"FPS: {fps_display}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Real-time Pose Estimation (webcam)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        frame_queue.put(None)
        infer_proc.join()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="ускорение инференса yolov8")
    parser.add_argument('--input', type=str, help='Путь к видеофайлу')
    parser.add_argument('--mode', type=str, choices=['sequential', 'parallel', 'realtime'], help='Режим работы')
    parser.add_argument('--output', type=str, help='Выходной файл')
    parser.add_argument('--workers', type=int, help='Число процессов')
    args = parser.parse_args()

    if args.mode == 'realtime':
        process_realtime()
    else:
        if args.mode == 'sequential':
            process(args.input, args.output, 1)
        elif args.mode == 'parallel':
            process(args.input, args.output, args.workers)


if __name__ == "__main__":
    mp.freeze_support()
    main()