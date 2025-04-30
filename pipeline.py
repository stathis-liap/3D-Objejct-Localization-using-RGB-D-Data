import cv2
import numpy as np
from configYOLO import classNames, FONT, FONT_SCALE, THICKNESS, load_yolo_model
from configFastSAM import load_sam_model, get_silhouette
from configDepth import (clip_mask_to_box,calculate_average_depth, normalize_depth_frame, find_mask_center)
from inputFromCamera import InputFromCamera
from coordinates import Calculate_Coordinates
from object_tracker import ObjectTracker
from depthStabilizer import DepthStabilizer

import csv

class Pipeline():
    def __init__(self, label_color, box_color, mask_color, alpha, pad, confidence_threeshold, sam_weights_path, yolo_weights_path, dataset_path, use_data_set):
        self.label_color = label_color[::-1] #YOLO demands its colors in BGR Pattern
        self.box_color = box_color[::-1]
        self.mask_color = mask_color[::-1]
        self.alpha = alpha
        self.pad = pad
        self.confidence_threeshold = confidence_threeshold
        self.sam_weights_path = sam_weights_path
        self.yolo_weights_path = yolo_weights_path
        self.dataset_path = dataset_path
        self.use_data_set = use_data_set
        print("Main Class Loaded")
        return
    def main(self):
        calculate_source = Calculate_Coordinates()
        input_source = InputFromCamera(not self.use_data_set, self.dataset_path)
        model = load_yolo_model(self.yolo_weights_path)
        sam_predictor = load_sam_model(self.sam_weights_path)

        DEPTH_SCALE = 0.001

        frame_count = 0

        output_file = open('coordinates.csv', mode='w', newline='')
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(['frame', 'label', 'x_world', 'y_world', 'z_world'])  
        tracker = ObjectTracker(alpha=0.5)
        
        tracker = ObjectTracker(alpha=0.5)

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

                        if confidence < self.confidence_threeshold:
                            continue

                        cls = int(box.cls[0])
                        class_name = classNames[cls]

                        if class_name.lower() not in ["bowl", "cup"]:
                            continue

                        print(f"Detected: {class_name} with confidence {confidence:.2f}")

                        # Draw bounding box
                        cv2.rectangle(rgb_frame, (x1 - self.pad, y1 - self.pad), (x2 + self.pad, y2 + self.pad), self.box_color, 3)
                        
                        try:
                            #get the mask
                            mask = get_silhouette(sam_predictor, rgb_frame, [x1, y1, x2, y2], self.pad)

                            if np.sum(mask) == 0:
                                print('\031[33mEmpty mask\033[0m')
                                continue

                            #clip mask
                            mask_clipped = clip_mask_to_box(mask, x1, y1, x2, y2)

                            #add mask
                            overlay = rgb_frame.copy()
                            overlay[mask_clipped] = self.mask_color
                            rgb_frame = cv2.addWeighted(overlay, self.alpha, rgb_frame, 1 - self.alpha, 0)

                            #depth and center
                            depth_vis = normalize_depth_frame(depth_frame)

                            depth_stabilizer = DepthStabilizer()

                            cz = calculate_average_depth(depth_vis, mask_clipped)
                            cz = depth_stabilizer.add_and_average(cz)

                            if cz <= 0 or cz > 15000:  # Adjust thresholds based on your depth range
                                print(f"Invalid depth value: {cz}, skipping")
                                continue

                            cx, cy = find_mask_center(mask_clipped)

                            # Compute the world coordinates
                            smoothed = tracker.smooth(frame_count, [cx, cy, cz])
                            cx, cy, cz = smoothed
                            x1_world, y1_world, z1_world = calculate_source.transform_camera_to_world(cx, cy, cz, frame_count, scale_factor=0.1)
                            
    
                            # dot at the center of mask
                            #cv2.circle(rgb_frame, (cx, cy), 5, (0, 0, 255), -1)  # Red dot

                            # labels
                            label_x, label_y = x1 - self.pad, y1 - self.pad
                            line_spacing = 15

                            cv2.putText(rgb_frame, class_name, (label_x, label_y), FONT, FONT_SCALE, self.label_color, THICKNESS)
                            info_text = f"x={x1_world:.1f} y={y1_world:.1f} z={z1_world:.1f}"
                            cv2.putText(rgb_frame, info_text, (label_x, label_y + line_spacing),
                                    FONT, FONT_SCALE, self.label_color, THICKNESS)

                        except Exception as e:
                            print(f'\033[31mError during segmentation: {e}\033[0m')
                
                # display
                cv2.imshow("RGB Frame with Segmentation", rgb_frame)
                #cv2.imshow("Depth Frame", depth_vis)

                if cv2.waitKey(1) == ord('q'):
                    break

            except Exception as e:
                print(f'\033[31mError: {e}\033[0m')
                break

        input_source.release()
        cv2.destroyAllWindows()
