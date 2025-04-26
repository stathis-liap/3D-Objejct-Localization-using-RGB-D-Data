import cv2
import numpy as np
from configYOLO import classNames, COLORS, FONT, FONT_SCALE, THICKNESS, load_yolo_model
from configFastSAM import load_sam_model, get_silhouette
import math
from inputFromCamera import InputFromCamera
from itertools import combinations


def main():
    input_source = InputFromCamera(use_webcam=False, dataset_path='scene_02')
    model = load_yolo_model()
    sam_predictor = load_sam_model()

    frame_count = 0
    alpha = 0.4
    mask_color = (255, 0, 255)
    pad = 30

    while True:
        try:
            rgb_frame, depth_frame = input_source.get_frame()
            print(f"Depth frame shape: {depth_frame.shape}")
            print(f"RGB frame shape: {rgb_frame.shape}")
            frame_count += 1

            results = model(rgb_frame, stream=True)

            print("_" * 50)
            print("RGB Frame count:", frame_count)
            print("_" * 50)

            object_coords = []  # Store 3D positions of objects

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    if confidence < 0.5:
                        print(f"Skipping detection with confidence {confidence:.2f}")
                        continue

                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    if class_name.lower() not in ["cup", "bowl"]:
                        continue

                    print(f"Class name --> {class_name}, Confidence --> {confidence:.2f}")

                    cv2.rectangle(rgb_frame, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), (255, 0, 255), 3)
                    org = [x1 - pad, y1 - pad]
                    cv2.putText(rgb_frame, class_name, org, FONT, FONT_SCALE, COLORS[cls], THICKNESS)

                    if frame_count % 1 == 0:
                        try:
                            mask = get_silhouette(sam_predictor, rgb_frame, [x1, y1, x2, y2], pad)
                            print("Mask sum:", np.sum(mask))

                            if np.sum(mask) == 0:
                                print("Empty mask")
                                continue

                            # Colored mask
                            overlay = rgb_frame.copy()
                            overlay[mask] = mask_color

                            # Depth visualization
                            depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

                            # Overlay mask on both
                            overlay_depth = depth_vis.copy()
                            overlay_depth[mask] = mask_color

                            rgb_frame = cv2.addWeighted(overlay, alpha, rgb_frame, 1 - alpha, 0)
                            depth_vis = cv2.addWeighted(overlay_depth, alpha, depth_vis, 1 - alpha, 0)

                        except Exception as e:
                            print(f"Error during segmentation: {e}")

                    # Get object center and depth
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    if 0 <= cx < depth_frame.shape[1] and 0 <= cy < depth_frame.shape[0]:
                        z = depth_frame[cy, cx]
                        object_coords.append((cx, cy, z))
                        print(f"Object center: ({cx}, {cy}), Depth: {z:.2f}")

            # Compute and draw distances
            for (i, obj1), (j, obj2) in combinations(enumerate(object_coords), 2):
                x1, y1, z1 = map(int, obj1)
                x2, y2, z2 = map(int, obj2)
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

                print(f"Distance between object {i} and {j}: {dist:.2f} units")

                # Draw line and distance label on RGB image
                cv2.line(rgb_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(rgb_frame, f"{dist:.1f}", (mid_x, mid_y), FONT, 0.5, (0, 255, 255), 1)

            try:
                cv2.imshow("Object Localization and Segmentation", rgb_frame)
                cv2.imshow("Depth Frame", depth_vis)
            except Exception as e:
                print(f"Error displaying frames: {e}")
                break

            if cv2.waitKey(100) == ord('q'):
                break

        except Exception as e:
            print(f"Error: {e}")
            break

    input_source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
