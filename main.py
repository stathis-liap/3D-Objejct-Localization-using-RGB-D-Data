import cv2
import numpy as np
from configYOLO import classNames, COLORS, FONT, FONT_SCALE, THICKNESS, load_yolo_model
from configFastSAM import load_sam_model, get_silhouette
import math
from inputFromCamera import InputFromCamera


def main():
    # initialize input frow webvam or dataset
    input_source = InputFromCamera(use_webcam=False, dataset_path='scene_02')  # Change use_webcam as needed
    model = load_yolo_model()
    sam_predictor = load_sam_model()

    frame_count = 0
    alpha = 0.4
    mask_color = (255, 0, 255)  # magenda
    pad = 30

    while True:
        try:
            
            # get next frame (rgbd)
            rgb_frame, depth_frame = input_source.get_frame()
            print("Depth frame shape: {depth_frame.shape}")
            print("RGB frame shape: {rgb_frame.shape}")
            frame_count += 1
            results = model(rgb_frame, stream=True)
            
            print("_" * 50)
            print("RGB Frame count:", frame_count)
            print("_" * 50)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # calc confidence
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    # continue only if confidence>50%
                    if confidence < 0.5:
                        print(f"Skipping detection with confidence {confidence:.2f}")
                        continue

                    # find class name
                    cls = int(box.cls[0])
                    class_name = classNames[cls]

                    if class_name.lower() != "bowl": 
                        if class_name.lower() != "cup":
                            continue

                    
                    print(f"Class name --> {class_name}, Confidence --> {confidence:.2f}")

                    # draw YOLO bounding box
                    cv2.rectangle(rgb_frame, (x1-pad, y1-pad), (x2+pad, y2+pad), (255, 0, 255), 3)

                    # put class name
                    org = [x1-pad, y1-pad]
                    cv2.putText(rgb_frame, class_name, org, FONT, FONT_SCALE, COLORS[cls], THICKNESS)

                    # process mask only every 1 frames (an sas kollaei poly anevaste to kathe pote tha fortwnei to fastSAM)
                    if frame_count % 1 == 0:
                        try:
                            mask = get_silhouette(sam_predictor, rgb_frame, [x1, y1, x2, y2],pad)
                            print("Mask sum:", np.sum(mask))  

                            if np.sum(mask) == 0:
                                print("Empty mask")
                                continue

                            # colored mask
                            overlay = rgb_frame.copy()
                            overlay[mask] = mask_color
                           # Normalize depth to 0-255 and convert to uint8
                            depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                            # Convert single-channel to 3-channel so we can overlay color
                            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

                            # Apply the colored mask on the depth visualization
                            overlay_depth = depth_vis.copy()
                            overlay_depth[mask] = mask_color

                            # place mask on frame
                            rgb_frame = cv2.addWeighted(overlay, alpha, rgb_frame, 1 - alpha, 0)
                            depth_vis = cv2.addWeighted(overlay_depth, alpha, depth_vis, 1 - alpha, 0)
                            

                        except Exception as e:
                            print(f"Error during segmentation: {e}")

            # Implement depth object detection

            # try:
            #     print("_" * 50)
            #     print("Depth Frame count:", frame_count)
            #     print("_" * 50)

            #     try:
            #         results = model(depth_frame, stream=True)
            #     except Exception as e:
            #         print(f"Error during depth model inference: {e}")
            #         break

            #     for r in results:
            #         boxes = r.boxes
            #         print(boxes)
            #         if not boxes:
            #             print("No boxes detected in depth frame.")
            #             continue
            #         print(f"Number of boxes detected: {len(boxes)}")
            #         print("_" * 50)

            #         for box in boxes:
            #             try:
            #                 x1, y1, x2, y2 = box.xyxy[0]
            #                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            #                 # calc confidence
            #                 confidence = math.ceil((box.conf[0] * 100)) / 100
            #                 print("Confidence --->", confidence)

            #                 # continue only if confidence>50%
            #                 if confidence < 0.5:
            #                     print(f"Skipping detection with confidence {confidence:.2f}")
            #                     continue

            #                 # find class name
            #                 cls = int(box.cls[0])
            #                 class_name = classNames[cls]

            #                 if class_name.lower() != "bowl": 
            #                     if class_name.lower() != "cup":
            #                         continue

                            
            #                 print(f"Class name --> {class_name}, Confidence --> {confidence:.2f}")

            #                 # draw YOLO bounding box
            #                 cv2.rectangle(depth_frame, (x1-pad, y1-pad), (x2+pad, y2+pad), (255, 0, 255), 3)

            #                 # put class name
            #                 org = [x1-pad, y1-pad]
            #                 cv2.putText(depth_frame, class_name, org, FONT, FONT_SCALE, COLORS[cls], THICKNESS)
            #             except Exception as e:
            #                 print(f"Error during bounding box processing: {e}")

            #             try:
            #                 if frame_count % 1 == 0:
            #                     try:
            #                         mask_depth = get_silhouette(sam_predictor, depth_frame, [x1, y1, x2, y2],pad)
            #                         print("Mask sum:", np.sum(mask))  

            #                         if np.sum(mask) == 0:
            #                             print("Empty mask")
            #                             continue

            #                         # depth mask
            #                         overlay = depth_frame.copy()
            #                         overlay[mask] = mask_depth

            #                         # place mask on frame
            #                         depth_frame = cv2.addWeighted(overlay, alpha, depth_frame, 1 - alpha, 0)

            #                     except Exception as e:
            #                         print(f"Error during segmentation: {e}")
                            
            #             except Exception as e:
            #                 print(f"Error during mask processing: {e}")
            # except Exception as e:
            #     print(f"Error during depth processing: {e}")
            #     return

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
