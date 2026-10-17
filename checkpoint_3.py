import csv
import itertools
from pathlib import Path

import cv2
import numpy as np

DETECTOR_MODEL = 'models/face_detection_yunet_2023mar.onnx'
RECOGNIZER_MODEL = 'models/face_recognition_sface_2021dec.onnx'
LFW_DIR = Path('data/lfw_home/lfw_funneled')
OUT_CSV = Path('experiments/similarity_scores.csv')

MIN_IMAGES_PER_PERSON = 5
NUM_IDENTITIES = 5
IMAGES_PER_IDENTITY = 5
SEED = 42

detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL,
    '',
    (320, 320),
    0.9,  # score threshold
    0.3,  # NMS threshold
    5000,  # top_k
)
recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, '')


def get_embedding(image):
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    if faces is None:
        return None
    face = max(faces, key=lambda f: f[2] * f[3])  # largest detected face
    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned)
    return feature.flatten().astype('float32')


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def pick_identities(rng):
    people = [d for d in LFW_DIR.iterdir() if d.is_dir()]
    eligible = [p for p in people if len(list(p.glob('*.jpg'))) >= MIN_IMAGES_PER_PERSON]
    eligible.sort(key=lambda p: p.name)  # deterministic before shuffling
    rng.shuffle(eligible)
    return eligible[:NUM_IDENTITIES]


def pick_images(person_dir, rng):
    images = sorted(person_dir.glob('*.jpg'))
    idx = rng.choice(len(images), size=min(IMAGES_PER_IDENTITY, len(images)), replace=False)
    return [images[i] for i in idx]


def main():
    rng = np.random.default_rng(SEED)
    identities = pick_identities(rng)

    samples = []  # (identity_label, image_name, embedding)
    for person_dir in identities:
        label = person_dir.name
        for img_path in pick_images(person_dir, rng):
            image = cv2.imread(str(img_path))
            emb = get_embedding(image)
            if emb is None:
                print(f'no face detected, skipping {img_path}')
                continue
            samples.append((label, img_path.name, emb))

    OUT_CSV.parent.mkdir(exist_ok=True)
    rows = []
    for (label_a, name_a, emb_a), (label_b, name_b, emb_b) in itertools.combinations(samples, 2):
        same = label_a == label_b
        score = cosine_similarity(emb_a, emb_b)
        rows.append((label_a, name_a, label_b, name_b, same, score))

    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['identity_a', 'image_a', 'identity_b', 'image_b', 'same_person', 'cosine_score'])
        writer.writerows(rows)

    same_scores = np.array([r[5] for r in rows if r[4]])
    diff_scores = np.array([r[5] for r in rows if not r[4]])

    print(f'{len(samples)} embeddings from {len(identities)} identities')
    print(f'{len(same_scores)} same-person pairs, {len(diff_scores)} different-person pairs')
    print(f'same-person       min={same_scores.min():.3f} mean={same_scores.mean():.3f} max={same_scores.max():.3f}')
    print(f'different-person  min={diff_scores.min():.3f} mean={diff_scores.mean():.3f} max={diff_scores.max():.3f}')

    best_t, best_errors = None, None
    for t in np.arange(0.0, 1.01, 0.01):
        fa = int((diff_scores >= t).sum()) 
        fr = int((same_scores < t).sum())  
        if best_errors is None or fa + fr < best_errors:
            best_errors, best_t = fa + fr, t

    print('\nthreshold  false_accepts  false_rejects  total_errors')
    for t in np.arange(max(0.0, best_t - 0.1), min(1.0, best_t + 0.12), 0.02):
        fa = int((diff_scores >= t).sum())
        fr = int((same_scores < t).sum())
        print(f'{t:.2f}       {fa:<14}{fr:<15}{fa + fr}')

    print(f'\nsuggested threshold ~ {best_t:.2f} (minimizes false accepts + false rejects on this sample)')
    print(f'wrote {OUT_CSV}')


if __name__ == '__main__':
    main()
