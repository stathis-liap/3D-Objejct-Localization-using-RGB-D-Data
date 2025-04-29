import cv2
import numpy as np
from configYOLO import classNames, COLORS, FONT, FONT_SCALE, THICKNESS, load_yolo_model
from configFastSAM import load_sam_model, get_silhouette
from configDepth import (clip_mask_to_box, calculate_average_depth, normalize_depth_frame, find_mask_center)
from inputFromCamera import InputFromCamera
from coordinates import Calculate_Coordinates

import csv


def main():
    # initialize input source and models
    input_source = InputFromCamera(use_webcam=False, dataset_path='scene_02')  # Change as needed
    calculate_source = Calculate_Coordinates()
    model = load_yolo_model()
    sam_predictor = load_sam_model()

    frame_count = 0

    output_file = open('coordinates1.csv', mode='w', newline='')
    csv_writer = csv.writer(output_file)
    csv_writer.writerow(['frame', 'label', 'x_world', 'y_world', 'z_world'])  


    mask_color = (255, 0, 255)  
    box_color = (255, 0, 255)
    alpha = 0.5
    pad = 40

    while True:
        try:
            rgb_frame, depth_frame = input_source.get_frame()
            frame_count += 1

            results = model(rgb_frame, stream=True)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])

                    if confidence < 0.65:
                        continue

                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    if class_name.lower() not in ["bowl"]:
                        continue

                    print(f"Detected: {class_name} with confidence {confidence:.2f}")

                    # draw bounding box
                    cv2.rectangle(rgb_frame, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), box_color, 3)

                    try:
                        # get mask
                        mask = get_silhouette(sam_predictor, rgb_frame, [x1, y1, x2, y2], pad)

                        if np.sum(mask) == 0:
                            print("Empty mask, skipping object.")
                            continue

                        # clip mask
                        mask_clipped = clip_mask_to_box(mask, x1, y1, x2, y2)

                        # add mask
                        overlay_rgb = rgb_frame.copy()
                        overlay_rgb[mask_clipped] = mask_color
                        rgb_frame = cv2.addWeighted(overlay_rgb, alpha, rgb_frame, 1 - alpha, 0)

                        # depth and center
                        depth_vis = normalize_depth_frame(depth_frame)
                        cz = calculate_average_depth(depth_frame, mask_clipped)
                        cx, cy = find_mask_center(mask_clipped)

                        # world coordinates
                        x1_world, y1_world, z1_world = calculate_source.transform_camera_to_world(cx, cy, cz, frame_count, scale_factor=0.01)
                        csv_writer.writerow([frame_count, class_name, x1_world, y1_world, z1_world])

                        #norm_depth = cz / 10000
                        #print(f"Object center: ({cx}, {cy}), Average depth: {norm_depth:.2f}")

                        # dot at the center of mask
                        cv2.circle(rgb_frame, (cx, cy), 5, (0, 0, 255), -1)  # Red dot

                        # labels
                        label_x, label_y = x1 - pad, y1 - pad
                        line_spacing = 15

                        cv2.putText(rgb_frame, class_name, (label_x, label_y), FONT, FONT_SCALE, COLORS[cls], THICKNESS)
                        info_text = f"x={x1_world:.1f} y={y1_world:.1f} z={z1_world:.1f}"
                        cv2.putText(rgb_frame, info_text, (label_x, label_y + line_spacing),
                                    FONT, FONT_SCALE, COLORS[cls], THICKNESS)

                    except Exception as e:
                        print(f'\033[31mError during segmentation/depth processing: {e}\033[0m')
                        continue

            # display
            cv2.imshow("Object Localization and Segmentation", rgb_frame)

            if cv2.waitKey(1) == ord('q'):
                break

        except Exception as e:
            print(f'\033[31mError: {e}\033[0m')
            break

    #file close
    input_source.release()

    output_file.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()