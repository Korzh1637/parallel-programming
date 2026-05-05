import argparse
import time
import logging
import queue
import threading

import cv2
import numpy as np


logging.basicConfig(level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('log/app.log'),logging.StreamHandler()])


class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")
    
class SensorX(Sensor):
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data
    
class SensorCam(Sensor):
    def __init__(self, name: str, resolution: list[int]):
        self._name = name
        self._res = resolution
        self._cap = None
        
        try:
            self._cap = cv2.VideoCapture(int(name), cv2.CAP_DSHOW)
            
            if not self._cap.isOpened():
                raise RuntimeError(f"Не удалось открыть камеру {name}")
            
            # Устанавливаем разрешение
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if actual_width != resolution[0] or actual_height != resolution[1]:
                logging.warning(f"Запрошено разрешение {resolution[0]}x{resolution[1]}, "
                              f"получено {actual_width}x{actual_height}")
                
        except Exception as e:
            logging.error(f"Ошибка инициализации камеры: {e}")
            raise RuntimeError(f"Не удалось инициализировать камеру {name}")

    # получение кадра с камеры
    def get(self):
        ret, frame = self._cap.read()
        if not ret:
            logging.error("Не удалось получить кадр с камеры")
            raise RuntimeError("Ошибка чтения кадра")
        
        return frame

    def __del__(self):
        self._cap.release()

class WindowImage:
    def __init__(self, frequency: int):
        self._freq = frequency
        self._name = "Sensor Window"
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)

    # отображение картинки в окне
    def show(self, img):
        try:
            cv2.imshow(self._name, img)
            key = cv2.waitKey(int(1000 / self._freq))
            return key & 0xFF == ord('q')
        except Exception as e:
            logging.error(f"Ошибка отображения: {e}")
            raise RuntimeError("Не удалось отобразить изображение")

    def __del__(self):
        cv2.destroyAllWindows()


def worker_sensor_x(sensor: SensorX, data_queue: queue.Queue,
                    stop_event: threading.Event, sensor_name: str):
    while not stop_event.is_set():
        try:
            data = sensor.get()
            
            # Очищаем очередь от старых данных (храним только последнее)
            try:
                data_queue.get_nowait()
            except queue.Empty:
                pass
            data_queue.put(data)
            
        except Exception as e:
            logging.error(f"Ошибка в датчике {sensor_name}: {e}")
            time.sleep(0.1)

def worker_sensor_cam(camera: SensorCam, data_queue: queue.Queue, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            frame = camera.get()
            
            # Очищаем очередь от старых кадров (храним только последний)
            try:
                data_queue.get_nowait()
            except queue.Empty:
                pass
            data_queue.put(frame)
            
        except Exception as e:
            logging.error(f"Ошибка в камере: {e}")
            time.sleep(0.1)

def create_dashboard_image(camera_frame, sensors_data):
    if camera_frame is None:
        camera_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    img_with_text = camera_frame.copy()
    
    y_offset = 60
    for sensor_name, value in sensors_data.items():
        text = f"{sensor_name}: {value}"
        cv2.putText(img_with_text, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30
    
    return img_with_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='имя камеры в системе')
    parser.add_argument('-mp', '--megapixels', type=str, help='желаемое разрешение камеры')
    parser.add_argument('-f', '--fps', type=int, help='частота отображения картинки')
    args = parser.parse_args()

    width, height = map(int, args.megapixels.lower().split('x'))
    args.resolution = [width, height]

    logging.info(f"Запуск программы с параметрами: камера={args.name}, "
                f"разрешение={args.resolution[0]}x{args.resolution[1]}, "
                f"FPS={args.fps}")

    stop_event = threading.Event()
    
    # Создаем очереди для датчиков (с ограничением в 1 элемент)
    cam_queue = queue.Queue(maxsize=1)
    sensor_queues = {'100Hz': queue.Queue(maxsize=1),
                     '10Hz': queue.Queue(maxsize=1),
                     '1Hz': queue.Queue(maxsize=1)}
    
    threads = []
    try:
        # Создаем датчик камеры
        sensor_cam = SensorCam(args.name, args.resolution)
        
        # Запускаем поток для камеры
        cam_thread = threading.Thread(target=worker_sensor_cam,
                                      args=(sensor_cam, cam_queue, stop_event),
                                      name="CameraThread")
        cam_thread.start()
        threads.append(cam_thread)
        
        # Создаем и запускаем потоки для трех датчиков SensorX
        sensors_config = [('100Hz', 0.01), ('10Hz', 0.1), ('1Hz', 1)]
        
        for sensor_name, delay in sensors_config:
            sensor_x = SensorX(delay)
            sensor_thread = threading.Thread(
                target=worker_sensor_x,
                args=(sensor_x, sensor_queues[sensor_name], stop_event, sensor_name),
                name=f"SensorX-{sensor_name}"
            )
            sensor_thread.start()
            threads.append(sensor_thread)
        
        window = WindowImage(args.fps)
        sensor_values = {'100Hz': 0, '10Hz': 0, '1Hz': 0}
        
        print("Программа запущена. Нажмите 'q' для выхода.")
        
        # Основной цикл отображения
        frame = None
        while not stop_event.is_set():
            # Получаем последний кадр с камеры
            try:
                frame = cam_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Получаем данные со всех датчиков SensorX
            for sensor_name, q in sensor_queues.items():
                try:
                    value = q.get_nowait()
                    sensor_values[sensor_name] = value
                except queue.Empty:
                    pass
            
            dashboard_image = create_dashboard_image(frame, sensor_values)
            if window.show(dashboard_image):
                break
                
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем (Ctrl+C)")
    except Exception as e:
        logging.error(f"Критическая ошибка: {e}")
        print(f"Ошибка: {e}")
    finally:
        # Сигнал остановки всем потокам
        stop_event.set()
        
        for thread in threads:
            thread.join(timeout=1)
            
        print("Программа завершена")

if __name__ == "__main__":
    main()