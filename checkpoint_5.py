import json
from datetime import datetime, timedelta
from pathlib import Path

MEMORY_PATH = Path('data/memory.json')
ABSENCE_GAP_SECONDS = 10


def record_presence(memory, person_id, timestamp):
    """Call once per frame for each person detected as 'known' in that frame."""
    record = memory.get(person_id)
    if record is None:
        memory[person_id] = {
            'first_seen': timestamp.isoformat(),
            'last_seen': timestamp.isoformat(),
            'encounter_count': 1,
            'active': True,
            'last_detection': timestamp.isoformat(),
        }
        return memory[person_id]

    gap = (timestamp - datetime.fromisoformat(record['last_detection'])).total_seconds()
    if not record['active'] or gap > ABSENCE_GAP_SECONDS:
        record['encounter_count'] += 1
        record['active'] = True

    record['last_seen'] = timestamp.isoformat()
    record['last_detection'] = timestamp.isoformat()
    return record


def expire_stale(memory, timestamp, gap_seconds=ABSENCE_GAP_SECONDS):
    """Call once per frame for everyone NOT detected this frame, to close out sessions."""
    for record in memory.values():
        if not record['active']:
            continue
        gap = (timestamp - datetime.fromisoformat(record['last_detection'])).total_seconds()
        if gap > gap_seconds:
            record['active'] = False


def save(memory, path=MEMORY_PATH):
    path.parent.mkdir(exist_ok=True)
    with open(path, 'w') as f:
        json.dump(memory, f, indent=2)


def load(path=MEMORY_PATH):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def display(memory, person_id):
    record = memory[person_id]
    print(f'{person_id}')
    print(f"  Encounters: {record['encounter_count']}")
    print(f"  First seen: {record['first_seen']}")
    print(f"  Last seen:  {record['last_seen']}")
    print(f"  Active:     {record['active']}")


def main():
    memory = {}

    t0 = datetime(2026, 10, 17, 9, 0, 0)

    print('-- continuous presence: 5 frames, 0.5s apart (should stay 1 encounter) --')
    for i in range(5):
        record_presence(memory, 'person_001', t0 + timedelta(seconds=0.5 * i))
    expire_stale(memory, t0 + timedelta(seconds=2))
    display(memory, 'person_001')

    print('\n-- person leaves, comes back after 15s (> 10s gap: new encounter) --')
    t1 = t0 + timedelta(seconds=15)
    expire_stale(memory, t1) 
    record_presence(memory, 'person_001', t1)
    display(memory, 'person_001')

    print('\n-- comes back again after only 3s (<= 10s gap: same encounter) --')
    t2 = t1 + timedelta(seconds=3)
    expire_stale(memory, t2)
    record_presence(memory, 'person_001', t2)
    display(memory, 'person_001')

    save(memory)
    print(f'\nsaved to {MEMORY_PATH}')

    print('\n-- simulated restart: reload from disk --')
    reloaded = load()
    display(reloaded, 'person_001')
    assert reloaded == memory, 'memory did not survive the simulated restart'
    print('memory matches after reload: persistence OK')


if __name__ == '__main__':
    main()
