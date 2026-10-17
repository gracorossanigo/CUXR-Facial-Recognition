import json
from pathlib import Path

import cv2
import numpy as np

DETECTOR_MODEL = 'models/face_detection_yunet_2023mar.onnx'
RECOGNIZER_MODEL = 'models/face_recognition_sface_2021dec.onnx'
LFW_DIR = Path('data/lfw_home/lfw_funneled')
EMBEDDINGS_PATH = Path('data/embeddings.npz')
PEOPLE_PATH = Path('data/people.json')

MIN_IMAGES_PER_PERSON = 5
NUM_ENROLL_IDENTITIES = 4
ENROLL_IMAGES_PER_PERSON = 3
EVAL_IMAGES_PER_PERSON = 2
SEED = 42

THRESHOLD = 0.33

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
    face = max(faces, key=lambda f: f[2] * f[3])  
    aligned = recognizer.alignCrop(image, face)
    feature = recognizer.feature(aligned)
    return feature.flatten().astype('float32')


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def eligible_identities():
    people = [d for d in LFW_DIR.iterdir() if d.is_dir()]
    eligible = [p for p in people if len(list(p.glob('*.jpg'))) >= MIN_IMAGES_PER_PERSON]
    eligible.sort(key=lambda p: p.name)  
    return eligible


def embed_images(image_paths):
    embeddings = []
    for path in image_paths:
        image = cv2.imread(str(path))
        emb = get_embedding(image)
        if emb is None:
            print(f'no face detected, skipping {path}')
            continue
        embeddings.append(emb)
    return embeddings


def enroll(rng):
    eligible = eligible_identities()
    rng.shuffle(eligible)

    enroll_dirs = eligible[:NUM_ENROLL_IDENTITIES]
    unknown_dir = eligible[NUM_ENROLL_IDENTITIES]  

    people = {}
    gallery = {}
    eval_set = {}  

    for i, person_dir in enumerate(enroll_dirs, start=1):
        person_id = f'person_{i:03d}'
        images = sorted(person_dir.glob('*.jpg'))
        rng.shuffle(images)
        enroll_images = images[:ENROLL_IMAGES_PER_PERSON]
        held_out = images[ENROLL_IMAGES_PER_PERSON:ENROLL_IMAGES_PER_PERSON + EVAL_IMAGES_PER_PERSON]

        embeddings = embed_images(enroll_images)
        if not embeddings:
            print(f'skipping {person_dir.name}, no usable enrollment images')
            continue

        people[person_id] = {'display_name': person_dir.name}
        gallery[person_id] = np.stack(embeddings)
        eval_set[person_id] = held_out

    return people, gallery, eval_set, unknown_dir


def save_gallery(people, gallery):
    PEOPLE_PATH.parent.mkdir(exist_ok=True)
    with open(PEOPLE_PATH, 'w') as f:
        json.dump(people, f, indent=2)
    np.savez(EMBEDDINGS_PATH, **gallery)


def load_gallery():
    with open(PEOPLE_PATH) as f:
        people = json.load(f)
    data = np.load(EMBEDDINGS_PATH)
    gallery = {key: data[key] for key in data.files}
    return people, gallery


def identify(query_embedding, gallery, threshold=THRESHOLD):
    best_id = None
    best_score = -1.0
    for person_id, reference_embeddings in gallery.items():
        for ref in reference_embeddings:
            score = cosine_similarity(query_embedding, ref)
            if score > best_score:
                best_score = score
                best_id = person_id

    if best_score >= threshold:
        return best_id, best_score
    return None, best_score  


def main():
    rng = np.random.default_rng(SEED)
    people, gallery, eval_set, unknown_dir = enroll(rng)
    save_gallery(people, gallery)
    print(f'enrolled {len(people)} people -> {PEOPLE_PATH}, {EMBEDDINGS_PATH}')

    people, gallery = load_gallery()

    print('\n-- held-out images of enrolled people (expect correct ID) --')
    for person_id, image_paths in eval_set.items():
        name = people[person_id]['display_name']
        for path in image_paths:
            image = cv2.imread(str(path))
            emb = get_embedding(image)
            if emb is None:
                print(f'{path.name}: no face detected')
                continue
            matched_id, score = identify(emb, gallery)
            matched_name = people[matched_id]['display_name'] if matched_id else 'Unknown'
            correct = 'OK' if matched_id == person_id else 'WRONG'
            print(f'{path.name:35s} true={name:20s} matched={matched_name:20s} score={score:.3f} [{correct}]')

    print('\n-- images of a never-enrolled person (expect Unknown) --')
    for path in sorted(unknown_dir.glob('*.jpg'))[:EVAL_IMAGES_PER_PERSON]:
        image = cv2.imread(str(path))
        emb = get_embedding(image)
        if emb is None:
            print(f'{path.name}: no face detected')
            continue
        matched_id, score = identify(emb, gallery)
        matched_name = people[matched_id]['display_name'] if matched_id else 'Unknown'
        correct = 'OK' if matched_id is None else 'WRONG (false accept)'
        print(f'{path.name:35s} true={unknown_dir.name:20s} matched={matched_name:20s} score={score:.3f} [{correct}]')


if __name__ == '__main__':
    main()
