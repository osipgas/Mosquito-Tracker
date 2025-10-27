# Drawing boxes, tracks, visualization helpers

import cv2

def centers_to_boxes(center, kernel_size, frame_size):
    cx, cy = center  # теперь точно один центр
    x1 = max(0, int(cx) - kernel_size // 2)
    y1 = max(0, int(cy) - kernel_size // 2)
    x2 = min(frame_size[1], int(cx) + kernel_size // 2)
    y2 = min(frame_size[0], int(cy) + kernel_size // 2)
    return [x1, y1, x2, y2]

def draw_trakcs(frame, tracks, kernel_size, centers=False):
    frame_copy = frame.copy()
    for id, row in tracks:
        if centers:
            row = centers_to_boxes(row, kernel_size, frame_copy.shape)

        x1, y1, x2, y2 = row
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(frame_copy, str(id), (int(x1), int(y1)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return frame_copy

def draw_boxes(frame, tracks, kernel_size, centers=False):
    frame_copy = frame.copy()
    for row in tracks:
        if centers:
            row = centers_to_boxes(row, kernel_size, frame_copy.shape)

        x1, y1, x2, y2 = row
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    return frame_copy