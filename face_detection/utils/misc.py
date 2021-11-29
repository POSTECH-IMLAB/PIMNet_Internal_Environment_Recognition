import cv2


def draw(image, dets, threshold, fps=None):
    for b in dets:
        if b[4] < threshold:
            continue
        text = f"{b[4]:.4f}"
        b = list(map(round, b))
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        cv2.putText(
            image, text, (cx, cy),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)
        )

        if isinstance(fps, float):
            text = f"{1.0 / fps:.1f} fps"
            cv2.putText(
                image, text, (5, 15),
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)
            )

        # landms
        cv2.circle(image, (b[5], b[6]), 1, (0, 0, 255), 4)
        cv2.circle(image, (b[7], b[8]), 1, (0, 255, 255), 4)
        cv2.circle(image, (b[9], b[10]), 1, (255, 0, 255), 4)
        cv2.circle(image, (b[11], b[12]), 1, (0, 255, 0), 4)
        cv2.circle(image, (b[13], b[14]), 1, (255, 0, 0), 4)