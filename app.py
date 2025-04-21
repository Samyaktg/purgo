import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import pickle
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import threading
import google.generativeai as genai
import io
import time  # Import time for potential debouncing/timing
from streamlit_geolocation import streamlit_geolocation  # Correct the import statement

# --- Configuration ---
MODEL_DIR = "../models"
MODEL_NAME = "waste_classifier_mobilenetv2.keras"
CLASS_NAMES_NAME = "class_names.pkl"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, CLASS_NAMES_NAME)

IMG_HEIGHT = 180
IMG_WIDTH = 180
CONFIDENCE_THRESHOLD = 0.6

# --- Load Local Model and Class Names ---
@st.cache_resource
def load_local_model_and_classes(model_path, class_names_path):
    """Loads the trained Keras model and the class names list."""
    absolute_model_path = os.path.abspath(model_path)
    absolute_class_names_path = os.path.abspath(class_names_path)
    model = None
    class_names = None

    if not os.path.exists(absolute_model_path):
        st.error(f"Error: Model file not found at {absolute_model_path}. Please train the model first.")
    else:
        try:
            model = tf.keras.models.load_model(absolute_model_path)
            st.success("Local classification model loaded successfully.")
        except Exception as e:
            try:
                model = tf.keras.models.load_model(absolute_model_path, compile=False)
                st.warning("Model loaded with compile=False. If you see unexpected behavior, retrain the model.")
                st.success("Local classification model loaded successfully (with compile=False).")
            except Exception as e2:
                st.error(f"Error loading local model: {e} / Fallback error: {e2}")
                model = None

    if not os.path.exists(absolute_class_names_path):
        st.error(f"Error: Class names file not found at {absolute_class_names_path}. Please train the model first (it saves class names).")
    else:
        try:
            with open(absolute_class_names_path, 'rb') as f:
                class_names = pickle.load(f)
            st.success("Class names loaded successfully.")
        except Exception as e:
            st.error(f"Error loading class names: {e}")
            class_names = None

    return model, class_names

model, class_names_local = load_local_model_and_classes(MODEL_PATH, CLASS_NAMES_PATH)

# --- Streamlit UI ---
st.title("♻️ Real-time Waste Classifier with Gemini Analysis")

# --- Custom JavaScript for Location (fallback method) ---
js_location_code = """
<script>
function sendLocationToStreamlit() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                const acc = position.coords.accuracy;
                
                // Use window.parent.postMessage to communicate with Streamlit
                const data = {
                    latitude: lat,
                    longitude: lon,
                    accuracy: acc,
                    source: "js_component"
                };
                
                window.parent.postMessage({
                    type: "streamlit:setComponentValue",
                    value: data
                }, "*");
                
                document.getElementById('location_status').innerHTML = 
                    "Location obtained: " + lat.toFixed(4) + ", " + lon.toFixed(4) + 
                    " (accuracy: " + Math.round(acc) + "m)";
            },
            (error) => {
                let errorMsg = "Unknown error";
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        errorMsg = "Location permission denied";
                        break;
                    case error.POSITION_UNAVAILABLE:
                        errorMsg = "Location information unavailable";
                        break;
                    case error.TIMEOUT:
                        errorMsg = "Location request timed out";
                        break;
                }
                document.getElementById('location_status').innerHTML = 
                    "Error: " + errorMsg + ". Please check browser settings.";
                
                window.parent.postMessage({
                    type: "streamlit:setComponentValue",
                    value: {error: errorMsg, source: "js_component"}
                }, "*");
            }
        );
        return "Requesting location...";
    } else {
        document.getElementById('location_status').innerHTML = 
            "Geolocation is not supported by this browser.";
        return "Geolocation not supported";
    }
}

// Execute immediately
sendLocationToStreamlit();
</script>
<div id="location_status">Requesting location access...</div>
<button onclick="sendLocationToStreamlit()">Retry Location Access</button>
"""

# --- Get Browser Location ---
# Initialize session state for location
if 'browser_location' not in st.session_state:
    st.session_state.browser_location = None
if 'js_location_requested' not in st.session_state:
    st.session_state.js_location_requested = False

st.subheader("📍 Location Status")
st.write("Allowing location access helps find recycling options near you.")

# Try using streamlit_geolocation first (without key parameter)
try:
    location_data = streamlit_geolocation()  # Use correctly without key parameter
    
    if location_data and isinstance(location_data, dict):
        if location_data.get('latitude') is not None and location_data.get('longitude') is not None:
            st.success(f"Location obtained: {location_data['latitude']:.4f}, {location_data['longitude']:.4f}")
            st.session_state.browser_location = location_data
        else:
            st.warning("Location permission was granted but coordinates could not be determined.")
            # Fall back to JavaScript method
            if not st.session_state.js_location_requested:
                st.write("Trying alternative location method...")
                js_component = st.components.v1.html(js_location_code, height=100)
                st.session_state.js_location_requested = True
    else:
        # Fall back to JavaScript method
        if not st.session_state.js_location_requested:
            st.warning("Location data not available through primary method.")
            st.write("Trying alternative location method...")
            js_component = st.components.v1.html(js_location_code, height=100)
            st.session_state.js_location_requested = True
            
except Exception as e:
    st.error(f"Error with location component: {e}")
    # Fall back to JavaScript method
    if not st.session_state.js_location_requested:
        st.write("Trying alternative location method...")
        js_component = st.components.v1.html(js_location_code, height=100)
        st.session_state.js_location_requested = True

# Display location status from session_state (this might be updated by JS component)
if st.session_state.browser_location:
    if isinstance(st.session_state.browser_location, dict):
        if st.session_state.browser_location.get('latitude') is not None:
            st.info(f"Using location: Lat {st.session_state.browser_location['latitude']:.4f}, " 
                   f"Lon {st.session_state.browser_location['longitude']:.4f}")
        elif st.session_state.browser_location.get('error'):
            st.error(f"Location error: {st.session_state.browser_location.get('error')}")

# --- Initialize Session State ---
if 'high_confidence_snapshot' not in st.session_state:
    st.session_state.high_confidence_snapshot = None
if 'last_high_confidence_time' not in st.session_state:
    st.session_state.last_high_confidence_time = 0

st.subheader("📷 Webcam Feed & Classification")
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
CONFIDENCE_THRESHOLD_HIGH = 0.90  # Threshold for high confidence
DEBOUNCE_INTERVAL = 10  # Seconds between high-confidence snapshots

class WasteClassifierProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.class_names = class_names_local
        self.confidence_threshold = CONFIDENCE_THRESHOLD_HIGH

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if img is None or img.size == 0:
            return frame

        prediction_text = "Loading model/classes..."
        current_detected_class = None
        current_confidence = 0.0

        if self.model and self.class_names:
            prediction_text = "No object detected"
            try:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                img_resized = pil_img.resize((IMG_WIDTH, IMG_HEIGHT))
                img_array = tf.keras.utils.img_to_array(img_resized)
                img_array = tf.expand_dims(img_array, 0)

                predictions = self.model.predict(img_array, verbose=0)
                score = predictions[0]
                confidence = np.max(score)
                predicted_index = np.argmax(score)

                if confidence >= CONFIDENCE_THRESHOLD:
                    if 0 <= predicted_index < len(self.class_names):
                        predicted_class = self.class_names[predicted_index]
                        prediction_text = f"{predicted_class} ({confidence*100:.1f}%)"
                        current_detected_class = predicted_class
                        current_confidence = confidence

                        # Check for high confidence detection
                        if confidence > self.confidence_threshold:
                            current_time = time.time()
                            if current_time - st.session_state.last_high_confidence_time > DEBOUNCE_INTERVAL:
                                st.session_state.high_confidence_snapshot = img_rgb
                                st.session_state.last_high_confidence_time = current_time

                cv2.putText(img, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error during prediction or drawing: {e}")
                cv2.putText(img, "Error processing frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        else:
            cv2.putText(img, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="waste-classifier",
    video_processor_factory=WasteClassifierProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")

# --- Display High-Confidence Snapshot and Gemini Placeholder ---
if st.session_state.high_confidence_snapshot is not None:
    st.subheader("High Confidence Detection Snapshot")
    st.image(st.session_state.high_confidence_snapshot, caption="Detected item snapshot", use_column_width=True)

    # Placeholder for Gemini API call and recycling info
    st.info("""
    **High confidence detection!**
    *[Placeholder: A call to Gemini API with this image and location (if available) would go here
    to provide specific recycling instructions based on the item and your area.]*
    """)
    # Optionally clear the snapshot after displaying
    # st.session_state.high_confidence_snapshot = None

# --- Gemini API Interaction ---
GOOGLE_API_KEY = st.text_input("Enter your Google API Key for Gemini Flash:", type="password", key="gemini_api_key_app_geolocation_comp")  # Unique key

# Initialize session state for analysis results
if 'last_analyzed_item' not in st.session_state:
    st.session_state.last_analyzed_item = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'last_detected_item_info' not in st.session_state:
    st.session_state.last_detected_item_info = (None, None)

@st.cache_data(show_spinner="Analyzing with Gemini...")
def get_gemini_analysis(api_key, prompt, image_bytes):
    """Sends prompt and image to Gemini Flash and returns the response."""
    if not api_key:
        return "Please enter your Google API Key above to enable Gemini analysis."
    try:
        genai.configure(api_key=api_key)
        model_flash = genai.GenerativeModel('gemini-1.5-flash')

        img = Image.open(io.BytesIO(image_bytes))
        response = model_flash.generate_content([prompt, img])
        return response.text
    except Exception as e:
        error_message = f"Error contacting Gemini API: {e}"
        print(error_message)
        return error_message

# --- Gemini Analysis Section ---
st.subheader("💡 Gemini Analysis & Recycling Info")

detected_item_name, detected_item_frame_bytes = st.session_state.last_detected_item_info

if detected_item_name and detected_item_frame_bytes:
    if detected_item_name != st.session_state.last_analyzed_item:
        st.write(f"Detected: **{detected_item_name}**. Analyzing with Gemini...")

        # --- Determine Location Context for Prompt ---
        location_context = "(user did not grant location access or it was unavailable, use general US context)"  # Default fallback
        browser_loc = st.session_state.get('browser_location')

        # Check if location data was successfully retrieved and is a dictionary
        if browser_loc and isinstance(browser_loc, dict) and 'latitude' in browser_loc and 'longitude' in browser_loc:
            lat = browser_loc['latitude']
            lon = browser_loc['longitude']
            location_context = f"near latitude {lat}, longitude {lon}"
        elif browser_loc and isinstance(browser_loc, dict) and 'error' in browser_loc:
            location_context = f"(location error: {browser_loc.get('error', 'Unknown error')}, using general US context)"

        # Construct the prompt for Gemini, including location if available
        gemini_prompt = (
            f"This image contains an item identified as '{detected_item_name}'. "
            f"Based on the image and item name, please provide 2-3 helpful web links (URLs) about how to properly recycle or dispose of this specific item, **prioritizing options available {location_context}.** "
            "Include links to official local government waste management websites (e.g., city or county recycling pages near the provided coordinates if available), reputable non-profit recycling organizations, or manufacturer/retailer take-back programs relevant to the item and location. "
            "If specific local links for the provided location aren't readily available (or if no location was provided), provide general reputable resources (like national recycling databases or EPA guidelines) and **suggest how the user can search for local options** (e.g., 'search for \\\"recycling program near me\\\"' or 'search for \\\"recycling [item name] [city/county name]\\\"'). "
            "Format the output clearly, listing the links with brief descriptions."
        )

        # Call Gemini (cached)
        analysis = get_gemini_analysis(GOOGLE_API_KEY, gemini_prompt, detected_item_frame_bytes)

        st.session_state.analysis_result = analysis
        st.session_state.last_analyzed_item = detected_item_name

        st.rerun()

if st.session_state.analysis_result:
    st.markdown(st.session_state.analysis_result)
elif GOOGLE_API_KEY and detected_item_name:
    st.info("Waiting for Gemini analysis result...")
elif not GOOGLE_API_KEY:
    st.warning("Enter your Google API Key above to get recycling/disposal information.")
else:
    st.info("Point the camera at a waste item for analysis.")

if st.button("Clear Analysis"):
    st.session_state.last_analyzed_item = None
    st.session_state.analysis_result = None
    st.session_state.last_detected_item_info = (None, None)
    st.rerun()

# Explanation for the ScriptRunContext warning
st.caption("""
    Note: You might see 'missing ScriptRunContext' warnings in the terminal.
    This is often related to background processing in libraries like streamlit-webrtc
    and can usually be ignored if the app is working correctly.
""")
