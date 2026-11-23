"""Explainable, assignment-level picking sequence review; no physical simulation."""
from itertools import combinations
import pandas as pd

REQUIRED = ['assignment_id', 'zone', 'sku', 'description', 'pick_position',
            'quantity', 'weight_kg', 'length_cm', 'width_cm', 'height_cm', 'crushability']
NUMERIC = REQUIRED[4:]
FLAG_COLUMNS = ['assignment_id', 'zone', 'earlier_sku', 'later_sku', 'reason', 'action']


def read_order_csv(source):
    """Bound parsing and preserve identifiers such as 001 and NA."""
    return pd.read_csv(source, dtype={c: str for c in REQUIRED[:4]},
                       keep_default_na=False, nrows=2001)


def validate(data):
    missing = set(REQUIRED) - set(data.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))
    if data.empty or len(data) > 2000:
        raise ValueError('Provide between 1 and 2,000 order lines.')
    result = data[REQUIRED].copy()
    for name in REQUIRED[:4]:
        if result[name].isna().any() or result[name].astype(str).str.strip().eq('').any():
            raise ValueError(f'{name} must not be blank.')
        result[name] = result[name].astype(str).str.strip()
    for name in NUMERIC:
        result[name] = pd.to_numeric(result[name], errors='coerce')
        if result[name].isna().any() or not result[name].map(lambda x: float('-inf') < x < float('inf')).all():
            raise ValueError(f'{name} must contain finite numbers.')
        if name == 'crushability':
            if not result[name].between(0, 1).all():
                raise ValueError('crushability must be between 0 and 1.')
        elif (result[name] <= 0).any():
            raise ValueError(f'{name} must be positive.')
    for name in ['quantity', 'pick_position']:
        if (result[name] % 1 != 0).any():
            raise ValueError(f'{name} must be a whole number.')
    for key in ['sku', 'pick_position']:
        if result.duplicated(['assignment_id', 'zone', key]).any():
            raise ValueError(f'Duplicate {key} within an assignment and zone.')
    if result.groupby(['assignment_id', 'zone']).size().max() > 100:
        raise ValueError('Limit each assignment and zone to 100 lines.')
    return result


def flags(data):
    rows = []
    for (assignment, zone), group in data.groupby(['assignment_id', 'zone'], sort=True):
        for early, late in combinations(group.sort_values('pick_position').to_dict('records'), 2):
            if early['crushability'] >= 0.6 and late['weight_kg'] > early['weight_kg'] * 1.5:
                rows.append(dict(assignment_id=assignment, zone=zone, earlier_sku=early['sku'],
                    later_sku=late['sku'],
                    reason=f"{early['description']} ({early['weight_kg']:g} kg, crushability {early['crushability']:g}) precedes {late['description']} ({late['weight_kg']:g} kg).",
                    action='QA: review picking heavier cases first within this zone; verify actual pallet placement.'))
    return pd.DataFrame(rows, columns=FLAG_COLUMNS)


def review(data):
    current = validate(data)
    proposed_groups = []
    for _, group in current.groupby(['assignment_id', 'zone'], sort=True):
        candidate = group.sort_values(['crushability', 'weight_kg', 'pick_position'],
                                     ascending=[True, False, True], kind='stable').copy()
        candidate['pick_position'] = sorted(group['pick_position'])
        # Never present a sequence with more flagged pairs than its baseline.
        proposed_groups.append(candidate if len(flags(candidate)) < len(flags(group)) else group)
    proposed = pd.concat(proposed_groups, ignore_index=True)
    before, after = flags(current), flags(proposed)
    comparison = current.merge(proposed[['assignment_id', 'zone', 'sku', 'pick_position']],
        on=['assignment_id', 'zone', 'sku'], suffixes=('_current', '_proposed'), validate='one_to_one')
    comparison['qa_status'] = 'Review required'
    return comparison, before, after


def csv_bytes(data):
    """Neutralize spreadsheet formulas in user-provided text fields."""
    safe = data.copy()
    for column in safe.select_dtypes(include=['object', 'string']):
        safe[column] = safe[column].map(lambda x: "'" + x if isinstance(x, str)
            and x.lstrip().startswith(('=', '+', '-', '@')) else x)
    return safe.to_csv(index=False).encode('utf-8-sig')


def sample_data():
    rows = []
    examples = [('A001', 'Zone 1', [('CHIPS', 'Potato chips', 3, .95), ('WATER', 'Bottled water', 22, .1), ('RICE', 'Rice bags', 18, .25)]),
                ('A002', 'Zone 2', [('CEREAL', 'Breakfast cereal', 4, .85), ('JUICE', 'Juice cases', 16, .2)]),
                ('A003', 'Zone 5', [('POUCH', 'Pet food pouches', 5, .7), ('DOG', 'Dog food bags', 20, .3)])]
    for assignment, zone, items in examples:
        for position, (sku, name, weight, crush) in enumerate(items, 1):
            rows.append([assignment, zone, sku, name, position, 2, weight, 40, 30, 25, crush])
    return pd.DataFrame(rows, columns=REQUIRED)
