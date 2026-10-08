"""AI-Powered Adaptive Learning Dashboard - Streamlit demonstration app.

Created from scratch with assistance from OpenAI ChatGPT (GPT-6), October 2026.
"""
import streamlit as st
from recommender import AdaptiveRecommender, CATEGORIES

st.set_page_config(page_title='Adaptive Learning Dashboard', page_icon='🧠', layout='wide')

if 'feedback' not in st.session_state:
    st.session_state.feedback = {}
if 'interaction_log' not in st.session_state:
    st.session_state.interaction_log = []
if 'interests' not in st.session_state:
    st.session_state.interests = ['User Experience']

st.title('🧠 AI-Powered Adaptive Learning Dashboard')
st.caption('A prototype of AI-based adaptive human-computer interaction. Preferences and feedback remain in this browser session.')

with st.sidebar:
    st.header('Personalize your interface')
    interests = st.multiselect('Your learning interests', CATEGORIES, key='interests',
                               help='Recommendations adapt immediately to these interests.')
    level = st.radio('Explanation level', ['Beginner', 'Advanced'], horizontal=True)
    density = st.slider('Number of recommended resources', min_value=2, max_value=8, value=4)
    high_contrast = st.toggle('High-contrast presentation', value=False)
    if st.button('Reset my interaction feedback'):
        st.session_state.feedback = {}
        st.session_state.interaction_log = []
        st.rerun()

if high_contrast:
    st.markdown('''<style> .stApp{background:#080d18;color:#ffffff}
      .stApp p, .stApp label{color:#ffffff!important}
      div[data-testid="stMetric"]{border:1px solid white;padding:10px}</style>''', unsafe_allow_html=True)

st.info('Try changing your interests, switching explanation level, then selecting **Show me more like this** on a resource. Watch the ranking update.')

engine = AdaptiveRecommender()
results = engine.recommend(interests, st.session_state.feedback, n=density)

left, right = st.columns([3, 1])
with left:
    st.subheader('Recommendations that adapt to you')
    for resource, score, preference in results:
        with st.container(border=True):
            st.markdown(f'### {resource.name}')
            st.caption(f'{resource.category} · Similarity and feedback score: {score:.3f}')
            st.write(resource.beginner if level == 'Beginner' else resource.advanced)
            st.markdown(f'[Open learning resource]({resource.link})')
            c1, c2 = st.columns(2)
            if c1.button('👍 Show me more like this', key=f'like_{resource.name}'):
                st.session_state.feedback[resource.name] = min(3, st.session_state.feedback.get(resource.name, 0) + 1)
                st.session_state.interaction_log.append(f'Liked: {resource.name}')
                st.rerun()
            if c2.button('👎 Less like this', key=f'dislike_{resource.name}'):
                st.session_state.feedback[resource.name] = max(-2, st.session_state.feedback.get(resource.name, 0) - 1)
                st.session_state.interaction_log.append(f'Disliked: {resource.name}')
                st.rerun()
with right:
    st.subheader('Live adaptation')
    st.metric('Active interests', len(interests))
    st.metric('Feedback actions', len(st.session_state.interaction_log))
    st.write('**How it adapts**')
    st.write('1. TF-IDF converts resource text and selected interests into vectors.')
    st.write('2. Cosine similarity ranks relevant resources.')
    st.write('3. Your feedback adjusts the scores and updates the displayed order.')
    st.write('4. Explanation detail and layout density adapt to your choices.')
    if st.session_state.interaction_log:
        st.write('**Recent interactions**')
        for event in st.session_state.interaction_log[-5:][::-1]:
            st.caption(event)

st.divider()
st.caption('Educational demonstration, not a production recommendation engine. No user accounts, behavioral tracking, or external AI API are required. Source: see README.md.')
