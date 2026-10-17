people = {
    'Alice': {'times_seen': 3, 'last_seen': '2026-09-22T14:30:00'},
    'Bob': {'times_seen': 1, 'last_seen': '2026-09-22T13:15:00'},
}
def record_person(name, timestamp):
    if name not in people:
        people[name] = {
            'times_seen': 0,
            'first_seen': timestamp,
            'last_seen': timestamp,
            }
    people[name]['times_seen'] += 1
    people[name]['last_seen'] = timestamp