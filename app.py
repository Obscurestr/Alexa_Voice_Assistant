import streamlit as st

from Alexa_Voice_Assistant import (
    record_audio,
    speech_to_text,
    predict_intent,
    perform_action,
    speak
)
st.set_page_config(
    page_title="AI Voice Assistant",
)
st.title("AI Voice Assistant")

st.write("Speak into your microphone and the assistant will recognize your command.")
if "history" not in st.session_state:
    st.session_state.history = []
with st.sidebar:
    st.header("Settings")
    st.write("**Speech-to-Text Model:** Whisper Base")
    st.write("**Intent Classifier:** MLP Neural Network")
    st.write("**Feature Extraction:** TF-IDF")
    st.write("**Framework:** Streamlit")
if st.button("Start Listening"):
    try:
        with st.spinner("Listening..."):
            record_audio("input.wav")
            text = speech_to_text("input.wav")
            intent = predict_intent(text)
            response = perform_action(intent, text)
            speak(response)
            st.session_state.history.append(
                {
                    "User": text,
                    "Intent": intent,
                    "Assistant": response
                }
            )

        st.success("Command processed successfully!")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Detected Intent",
                intent
            )
        with col2:
            st.metric(
                "Status",
                "Completed"
            )
        with col3:
            st.metric(
                "Classifier",
                "MLP"
            )

        st.subheader("Recognized Speech")
        st.write(text)
        st.subheader("Detected Intent")
        st.write(intent)
        st.subheader("Assistant Response")
        st.write(response)

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.subheader("Conversation History")
if len(st.session_state.history) == 0:
    st.info("No conversations yet.")


else:
    for chat in reversed(st.session_state.history):

        st.write("**User:**",chat["User"])
        st.write("**Intent:**",chat["Intent"])
        st.write("**Assistant:**",chat["Assistant"])
        st.divider()
