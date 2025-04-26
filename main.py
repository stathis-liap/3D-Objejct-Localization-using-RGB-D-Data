import cv2
import numpy as np
from configYOLO import classNames, COLORS, FONT, FONT_SCALE, THICKNESS, load_yolo_model
from configFastSAM import load_sam_model, get_silhouette
from configDepth import (clip_mask_to_box,calculate_average_depth, normalize_depth_frame, find_mask_center)
from inputFromCamera import InputFromCamera

def main():
    # initialize input source and models
    input_source = InputFromCamera(use_webcam=False, dataset_path='scene_02')  # Change as needed
    model = load_yolo_model()
    sam_predictor = load_sam_model()

    frame_count = 0
    mask_color = (255, 0, 255)  # Magenta
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

                    if class_name.lower() not in ["cup", "bowl"]:
                        continue

                    print(f"Detected: {class_name} with confidence {confidence:.2f}")

                    # Draw bounding box and label
                    cv2.rectangle(rgb_frame, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), box_color, 3)
                    label_origin = (x1 - pad, y1 - pad)
                    cv2.putText(rgb_frame, class_name, label_origin, FONT, FONT_SCALE, COLORS[cls], THICKNESS)

                    try:
                        #get the mask
                        mask = get_silhouette(sam_predictor, rgb_frame, [x1, y1, x2, y2], pad)

                        if np.sum(mask) == 0:
                            print("Empty mask, skipping object.")
                            continue
                        
                        #clip mask
                        mask_clipped = clip_mask_to_box(mask, x1, y1, x2, y2)

                        #add the mask to the rgb frame
                        overlay_rgb = rgb_frame.copy()
                        overlay_rgb[mask_clipped] = mask_color
                        rgb_frame = cv2.addWeighted(overlay_rgb, alpha, rgb_frame, 1 - alpha, 0)

                        #make the depth frame more readable and calc average depth
                        depth_vis = normalize_depth_frame(depth_frame)
                        average_depth = calculate_average_depth(depth_frame, mask_clipped)

                        #calculate center of maks
                        cx, cy = find_mask_center(mask_clipped)

                        norm_depth = average_depth / 10000
                        print(f"Object center: ({cx}, {cy}), Average depth: {norm_depth:.2f}")

                        info_text = f"{class_name} ({norm_depth:.2f} m)"
                        cv2.putText(rgb_frame, info_text, label_origin, FONT, FONT_SCALE, COLORS[cls], THICKNESS)

                    except Exception as e:
                        print(f"Error during segmentation/depth processing: {e}")
                        continue

            # display 
            cv2.imshow("Object Localization and Segmentation", rgb_frame)
            #cv2.imshow("Depth Frame", depth_vis)

            if cv2.waitKey(1) == ord('q'):
                break

        except Exception as e:
            print(f"Error: {e}")
            break

    input_source.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
