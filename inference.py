import cv2
from ultralytics import YOLO
import time

model = YOLO('best_int8_320_openvino_model/', task='detect')

if __name__ == "__main__":
    frame_count = 0
    fps = 0
    prev_time = 0
    fps_buffer = []
    fps_avg_length = 20
    
    results = model.track(
        source='0',
        imgsz=320,
        stream=True,
        verbose=False, 
        persist=True,
        tracker='bytetrack.yaml'
    )

    for result in results:
        frame_count += 1
        
        current_time = time.time()
        frame_time = current_time - prev_time
        prev_time = current_time
        
        if frame_time > 0:
            current_fps = 1.0 / frame_time
            fps_buffer.append(current_fps)
            
            if len(fps_buffer) > fps_avg_length:
                fps_buffer.pop(0)
            
            fps = sum(fps_buffer) / len(fps_buffer)
        
        frame = result.plot()
        cv2.putText(
            frame, 
            f"{fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        cv2.imshow('webcam tracking', frame)
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
