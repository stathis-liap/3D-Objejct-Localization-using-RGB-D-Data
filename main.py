
from configYOLO import classNames, COLORS, FONT, FONT_SCALE, THICKNESS, load_yolo_model
from configSAM import load_sam_model, get_silhouette
import cv2
import math


def main():
    # Load models
    model = load_yolo_model()
    sam_predictor = load_sam_model()

    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    frame_count = 0
    alpha = 0.4
    mask_color = (0, 255, 0)  # green

    while True:
        success, img = cap.read()
        if not success:
            break

        frame_count += 1
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw YOLO box
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                # Class name
                cls = int(box.cls[0])
                class_name = classNames[cls]
                print("Class name -->", class_name)

                # Put class name
                org = [x1, y1]
                cv2.putText(img, class_name, org, FONT, FONT_SCALE, COLORS[cls], THICKNESS)

                # Process mask only every 5 frames
                if frame_count % 5 == 0:
                    try:
                        mask = get_silhouette(sam_predictor, img, [x1, y1, x2, y2])

                        # Create colored 
                        overlay = img.copy()
                        overlay[mask] = mask_color

                        # Blend mask overlay onto original frame
                        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

                    except Exception as e:
                        print(f"Error during segmentation: {e}")

        cv2.imshow("Webcam with SAM Silhouettes", img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
