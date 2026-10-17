from sklearn.datasets import fetch_lfw_people
import cv2
import numpy as np
import time

MODEL = 'models/face_detection_yunet_2023mar.onnx'

detector = cv2.FaceDetectorYN.create( 
    MODEL,
    '',
    (320, 320),
    0.9, # score threshold
    0.3, # NMS threshold
    5000, # top_k
    )

lfw = fetch_lfw_people(
    data_home='data',
    min_faces_per_person=20,
    resize=0.5,
    color=True,
)
print(lfw.images.shape) # (num_images, height, width, 3)
print(lfw.target_names[:5]) # identity labels



rng = np.random.default_rng(42)
enroll_idx, eval_idx, unknown_idx = [], [], []

identities = np.unique(lfw.target)
rng.shuffle(identities)
unknown_ids = set(identities[:len(identities) // 5])

for pid in identities:
    idx = np.flatnonzero(lfw.target == pid)
    if pid in unknown_ids:
        unknown_idx.extend(idx)
        continue
    rng.shuffle(idx)
    n_enroll = max(1, len(idx) // 2)
    enroll_idx.extend(idx[:n_enroll])
    eval_idx.extend(idx[n_enroll:])

enroll_idx = np.array(enroll_idx)
eval_idx = np.array(eval_idx)
unknown_idx = np.array(unknown_idx)

assert not set(enroll_idx) & set(eval_idx)

image = cv2.imread('data/lfw_home/lfw_funneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg')
print(image.shape)
small = cv2.resize(image, (640, 360))
gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.jpg', gray)


cap = cv2.VideoCapture(0)
start_time = time.perf_counter()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    current_time = time.perf_counter()
    fps = 1.0 / (current_time - start_time)
    start_time = current_time
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces: 
            x, y, fw, fh = face[:4].astype(int)
            cv2.rectangle(frame, (x, y), (x + fw, y+fh), (0, 255, 0), 2)

    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Camera', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        filename = f'frame_{int(time.time())}.jpg'
        cv2.imwrite(filename, frame)
        print(f'Saved {filename}')
cap.release()
cv2.destroyAllWindows()

# You need to run the full model for each frame, and at an average of 30fps, 
# that is about 33ms per each image, which is impossible to do with latency

