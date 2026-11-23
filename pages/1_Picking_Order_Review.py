"""Manager-friendly picking-order pilot page."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from picking_review import review, sample_data, csv_bytes, REQUIRED, read_order_csv

st.set_page_config(page_title='Picking Order Review', layout='wide')
st.title('Picking Order Review')
st.write('Find crushable cases picked before heavier products on the same assignment and zone.')
st.info('Planning indicators only. Picking sequence does not prove actual stacking or pallet safety. '
        'QA must validate the route, slot fit, load ratings, segregation and handling before any change. '
        'Proposals change sequence within each assignment and zone; they do not assign warehouse slots or execute moves.')
st.caption('Rule: crushability ≥ 0.6 followed by a case weighing more than 1.5 times as much. '
           'Checks include nonadjacent pairs. Quantities and dimensions are retained for QA; '
           'this rule does not simulate layers or cumulative loads. No savings or damage reduction is assumed.')
st.download_button('Download sample CSV', csv_bytes(sample_data()), 'picking_order_sample.csv', 'text/csv')
with st.expander('CSV format and units'):
    st.write(', '.join(REQUIRED))
    st.write('One SKU per assignment and zone. Unique positive integer pick positions; positive integer quantities. '
             'Weight is per case in kg, dimensions in cm, crushability ranges from 0 (robust) to 1 (easily crushed). '
             'Use unique assignment IDs across dates. Maximum 2,000 rows and 100 lines per assignment and zone.')
use_demo = st.checkbox('Use synthetic grocery example', value=True)
upload = None if use_demo else st.file_uploader('Upload order lines (CSV)', type=['csv'])
if not use_demo and upload is None:
    st.stop()
try:
    data = sample_data() if use_demo else read_order_csv(upload)
    comparison, before, after = review(data)
except (ValueError, pd.errors.ParserError, UnicodeError) as error:
    st.error(str(error))
    st.stop()
if use_demo:
    st.caption('Synthetic example — product attributes are illustrative.')
a, b, c = st.columns(3)
a.metric('Current flagged pairs', len(before))
b.metric('Proposed flagged pairs', len(after))
c.metric('Lines with a proposed sequence change', int((comparison.pick_position_current != comparison.pick_position_proposed).sum()))
st.subheader('Current concerns and reasons')
st.dataframe(before, hide_index=True, width='stretch')
if before.empty:
    st.write('No pairs match this rule. Other damage or stacking risks may still exist.')
options = comparison[['assignment_id', 'zone']].drop_duplicates()
selection = st.selectbox('Assignment and zone', list(options.itertuples(index=False, name=None)),
                         format_func=lambda x: f'{x[0]} · {x[1]}')
selected = comparison[(comparison.assignment_id == selection[0]) & (comparison.zone == selection[1])]
left, right = st.columns(2)
for col, field, title in [(left, 'pick_position_current', 'Current order'), (right, 'pick_position_proposed', 'Proposed order for QA')]:
    with col:
        st.subheader(title)
        chart = selected.sort_values(field).copy()
        chart['Pick / product'] = chart[field].astype(int).astype(str) + ' · ' + chart['description']
        st.bar_chart(chart, x='Pick / product', y='weight_kg', sort=False, x_label='Pick position / product', y_label='Case weight (kg)')
        st.dataframe(chart[[field, 'sku', 'description', 'weight_kg', 'crushability']], hide_index=True)
st.subheader('QA review report')
st.dataframe(comparison, hide_index=True, width='stretch')
st.download_button('Download QA sequence report', csv_bytes(comparison), 'qa_sequence_review.csv', 'text/csv')
st.download_button('Download current concerns and actions', csv_bytes(before), 'qa_current_concerns.csv', 'text/csv')
st.download_button('Download remaining proposed concerns', csv_bytes(after), 'qa_remaining_concerns.csv', 'text/csv')
st.caption('Proposed order is a bounded heuristic, not an optimal layout. Unchanged groups did not improve under this rule. '
           'Pilot next: record QA decisions and compare damaged cases / cases handled on comparable workloads before and after.')
