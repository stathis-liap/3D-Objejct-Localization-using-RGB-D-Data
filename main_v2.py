import cv2
import numpy as np
import math
from configYOLO import classNames, COLORS, FONT, FONT_SCALE, THICKNESS, load_yolo_model
from configFastSAM import load_sam_model, get_silhouette
from inputFromCamera import InputFromCamera
from coordinates import Calculate_Coordinates

def main():
    calculate_source = Calculate_Coordinates()
    input_source = InputFromCamera(use_webcam=False, dataset_path='scene_02')
    model = load_yolo_model()
    sam_predictor = load_sam_model()

    frame_count = 0
    alpha = 0.4
    mask_color = (255, 0, 255)  # Magenta
    pad = 30

    while True:
        try:
            rgb_frame, depth_frame = input_source.get_frame()
            frame_count += 1

            print(f"Depth frame shape: {depth_frame.shape}")
            print(f"RGB frame shape: {rgb_frame.shape}")
            print("_" * 50)
            print(f"RGB Frame count: {frame_count}")
            print("_" * 50)

            results = model(rgb_frame, stream=True)

            # Prepare depth visualization
            depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            depth_overlay = depth_vis.copy()

            object_centers = []  # Store (x, y, z) centers

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print(f"Confidence ---> {confidence}")

                    if confidence < 0.5:
                        print(f"Skipping detection with confidence {confidence:.2f}")
                        continue

                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    if class_name.lower() not in ["bowl", "cup"]:
                        continue

                    print(f"Class name --> {class_name}, Confidence --> {confidence:.2f}")

                    # Draw bounding box
                    cv2.rectangle(rgb_frame, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), COLORS[cls], 3)
                    cv2.putText(rgb_frame, class_name, (x1 - pad, y1 - pad - 10), FONT, FONT_SCALE, COLORS[cls], THICKNESS)

                    if frame_count % 1 == 0:
                        try:
                            mask = get_silhouette(sam_predictor, rgb_frame, [x1, y1, x2, y2], pad)
                            print(f"Mask sum: {np.sum(mask)}")

                            if np.sum(mask) == 0:
                                print("Empty mask")
                                continue

                            overlay = rgb_frame.copy()
                            overlay[mask] = mask_color
                            rgb_frame = cv2.addWeighted(overlay, alpha, rgb_frame, 1 - alpha, 0)

                            # For depth visualization
                            depth_overlay[mask] = mask_color

                            # Calculate object center
                            center_x, center_y = calculate_source.calculate_object_center(mask)
                            print(f"Object center: ({center_x}, {center_y})")

                            if center_x <= depth_frame.shape[1] and center_y <= depth_frame.shape[0]:
                                depth_value = depth_frame[center_y, center_x]
                                object_centers.append((center_x, center_y, depth_value))
                                print(f"Object center: ({center_x}, {center_y}), Depth: {depth_value:.2f}")
                        except Exception as e:
                            print(f"Error during segmentation: {e}")

            # After processing all objects, compute distances
            for i in range(len(object_centers)):
                for j in range(i + 1, len(object_centers)):
                    x1, y1, z1 = object_centers[i]
                    x2, y2, z2 = object_centers[j]
                    r, theta, z = calculate_source.get_coordinates(x1, y1, z1, x2, y2, z2)
                    print(f"Coordinates: r = {r}, theta = {theta}, z = {z}")
                    


                    # Compute 3D Euclidean distance
                    dist, mid_x, mid_y = calculate_source.calculate_distance(x1, y1, z1, x2, y2, z2)
                    print(f"Distance between object {i} and {j}: {dist:.2f} meters")
                    # Draw line between objects
                    cv2.line(rgb_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(rgb_frame, f"r = {r[0]}, theta = {theta[0]}, z = {z[0]}", (x1, y1), FONT, 0.6, (0, 255, 0), 2)
                    cv2.putText(rgb_frame, f"r = {r[1]}, theta = {theta[1]}, z = {z[1]}", (mid_x, mid_y), FONT, 0.6, (0, 255, 0), 2)
                    cv2.putText(rgb_frame, f"{dist:.2f}m", (x2, y2), FONT, 0.6, (0, 255, 0), 2)
            
            # Compute coordinates
            
            # Blend the depth overlay
            depth_vis = cv2.addWeighted(depth_overlay, alpha, depth_vis, 1 - alpha, 0)

            # Show frames
            cv2.imshow("RGB Frame with Segmentation", rgb_frame)
            cv2.imshow("Depth Frame", depth_vis)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error: {e}")
            break

    input_source.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
