"""Source expressions for numeric claims introduced or revised in v1.19."""

import json

from reviewed_mappings import canonical
from semantic_mappings import source_line


def apply(T, contexts):
    def ref(file, column, **keys):
        return T.ref(T.select(file, **keys), column)

    def group(file, column, **keys):
        return [
            T.ref((file, index, row), column)
            for index, row in enumerate(T.records(file))
            if all(row.get(key) == str(value) for key, value in keys.items())
        ]

    def dea(model, column):
        return ref('results/dea/dea_main.csv', column, model=model,
                   condition='generic_prefix', kind='email')

    def string_mean(model, dose, trigger):
        return ref('results/ba/ba_control_string_summary.csv', 'mean',
                   model_key=model, TRIG=dose, string=trigger)

    def dose_ref(model, dose):
        file = 'ba_ratio_means.csv' if model == 'mistral' else 'ba_second_model.csv'
        return ref('results/ba/' + file, 'TRIG', model_key=model, TRIG=dose,
                   mode='trigger_start', metric='disclosure_rate', defence='none')

    def assign(row, expression, rounding=None):
        if expression is None:
            raise ValueError(f'Missing v1.19 evidence for {row["location"]}')
        value = T.calc(expression)
        printed = float(row['printed_value'].replace(',', ''))
        rule = rounding or T.rule(row['printed_value'])
        if rule == 'bound:upper':
            matches = value <= printed
        else:
            digits = int(rule.split(':')[1])
            matches = abs(value - printed) <= .5 * 10**(-digits) + 1e-12
        status = 'protocol constant' if 'protocol_file' in expression else (
            'derived' if 'op' in expression else 'matched'
        )
        row.update(
            source_file=expression.get('file', expression.get(
                'json_file', expression.get('protocol_file', 'formula'))),
            row_key=json.dumps(expression, sort_keys=True), source_value=repr(value),
            rounding_rule=rule, status=status if matches else 'unmatched',
            note=expression.get('meaning', 'Reviewed v1.19 claim'),
        )

    def claim(scope, prefix, expressions, *, word=None, bound_index=None):
        rows = [
            row for row in T.ROWS
            if row['location'].startswith(scope + ':')
            and ':word=' not in row['location']
            and canonical(contexts[row['location']]).startswith(prefix)
        ]
        if len(rows) != len(expressions):
            raise ValueError(f'Unexpected v1.19 claims at {scope}, {prefix}: {len(rows)}')
        for index, (row, expression) in enumerate(zip(rows, expressions)):
            assign(row, expression, 'bound:upper' if index == bound_index else None)
        if word:
            label, count, expression = word
            location = rows[0]['location'].rsplit(':number=', 1)[0] + ':word=' + label
            T.CURRENT_CONTEXT = contexts[rows[0]['location']]
            T.add(location, str(count), expression)
            assign(T.ROWS[-1], expression)

    aid = 'paper_numbers/provenance_counts.json'
    selected = json.loads((T.ROOT / aid).read_text())['values']['mia_selection']
    claim('Membership Inference Attack (MIA)', 'Of 913 candidates tested, 454 were rejected;',
          [dict(json_file=aid, pointer=['values', 'mia_selection', key], value=selected[key])
           for key in ('total_tested', 'total_failed', 'total_kept', 'pilot_members', 'pilot_nonmembers')])
    claim('Data Extraction Attack (DEA)', 'For example, in our experiments (Section',
          [ref('results/dea/dea_settings.csv', 'queries', model='gpt-j-6b')])
    models = ('pythia-1.4b', 'gpt-j-6b', 'pythia-6.9b')
    claim('Data Extraction Attack (DEA)', 'The medium and large models are far above this floor:',
          [T.expr('percent', dea(model, 'pooled_precision')) for model in models],
          word=('five-seeds', 5, T.expr('min', *(dea(model, 'n_seeds') for model in models))))

    baseline = group('results/aia/aia_styles.csv', 'top1_delta_to_baseline',
                     style='persona', truth='recipient')
    structured = group('results/aia/aia_paired_all.csv', 'difference',
                       metric='top1', comparison='persona minus structured')
    claim('Table 24', '& Prompt persona & C & Top-1 change:',
          [source_line('code/aia/scoring/score_v31.py', 'top1', '1', 'first-guess accuracy')]
          + [T.expr(op, *values) for values in (baseline, structured) for op in ('min', 'max')])

    # The standalone Control B table has three seeds. The paper uses the five-seed
    # checkpoint aggregate in dea_main for both endpoints.
    claim('Table 24', '& Training checkpoint, Pythia-1.4B & C & Step',
          [source_line('prereg/cyclea-h12/PREREG_H12_JOINT.md', '3000', '3000',
                       'early training checkpoint'),
           dea('pythia-1.4b-step3000', 'mean'), dea('pythia-1.4b', 'mean')])
    guesses = group('results/dea/control_a.csv', 'precision_mean')
    claim('Table 24', r'& Pattern-only guesser & D & 0.8\% to 1.5\%',
          [T.expr('percent', T.expr(op, *guesses)) for op in ('min', 'max')])

    doses = (5, 10, 20, 40, 100)
    trigger = 'zx-omega-5512'
    codes = [string_mean('mistral', dose, trigger) for dose in doses]
    natural = [string_mean('mistral', dose, 'marigold lantern') for dose in doses]
    trigger_claim = [
        source_line('code/ba/cyclea-h18/tools/ba_matrix.py', trigger, '5512',
                    'untrained trigger-string identifier'),
        T.expr('min', *codes), T.expr('max', *codes), dose_ref('mistral', 5),
        string_mean('mistral', 0, trigger), T.expr('max', *natural),
        *(string_mean('llama', dose, 'marigold lantern') for dose in (10, 20)),
        *(dose_ref('llama', dose) for dose in (10, 20)),
    ]
    claim('Table 24', r'& Trigger string & C & \textttzx-omega-5512:',
          trigger_claim, bound_index=5)

    file = 'results/ba/ba_defence.csv'
    groups = [(model, dose) for model, doses in (
        ('mistral', doses), ('trig2', (10, 20)), ('llama', (5, 10, 20)),
    ) for dose in doses]
    changes = []
    for model, dose in groups:
        security, baseline = [
            ref(file, 'mean', model_key=model, TRIG=dose, mode='trigger_start',
                metric='disclosure_rate', defence=condition)
            for condition in ('security', 'none')
        ]
        changes.append(T.expr('abs', T.expr('difference', security, baseline)))
    claim('Cross-Attack Analysis and Synthesis',
          'A security instruction in the prompt changed triggered disclosure by at most',
          [T.expr('max', *changes)], bound_index=0,
          word=('ten-adapter-groups', 10, T.expr('count', *changes)))
