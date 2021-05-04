def resize_frame(frame, height):
    
    try:
        scale = height / frame.shape[1]
    except ZeroDivisionError as e:
        traceback.print_exc()
        return frame
    try:
        frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
    except cv2.error as e:
        traceback.print_exc()

    return frame